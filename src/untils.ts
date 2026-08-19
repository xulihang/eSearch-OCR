let canvas = (width: number, height: number) => {
    if (typeof OffscreenCanvas !== 'undefined') {
        return new OffscreenCanvas(width, height);
    }
    // Fallback for browsers without OffscreenCanvas (e.g. iOS < 16.4)
    const c = document.createElement('canvas');
    c.width = width;
    c.height = height;
    return c;
};

export function newCanvas(width: number, height: number) {
    return canvas(width, height);
}

export function setCanvas(x) {
    canvas = x;
}

export function int(num: number) {
    return num > 0 ? Math.floor(num) : Math.ceil(num);
}
export function clip(n: number, min: number, max: number) {
    return Math.max(min, Math.min(n, max));
}
/**
 *
 * @param  data 原图
 * @param  w 输出宽
 * @param  h 输出高
 * @param  fill 小于输出宽高的部分填充还是拉伸
 */
export function resizeImg(
    data: ImageData,
    w: number,
    h: number,
    fill?: "fill",
    smoothing: false | "low" | "medium" | "high" = "high",
) {
    const ctx = resizeImgC(data, w, h, fill, smoothing);
    return ctx.getImageData(0, 0, w, h);
}

/**
 *
 * @param  data 原图
 * @param  w 输出宽
 * @param  h 输出高
 * @param  fill 小于输出宽高的部分填充还是拉伸
 */
export function resizeImgC(
    data: ImageData,
    w: number,
    h: number,
    fill?: "fill",
    smoothing: false | "low" | "medium" | "high" = "high",
) {
    const x = data2canvas(data);
    const src = newCanvas(w, h);
    const ctx = src.getContext("2d")!;
    ctx.imageSmoothingEnabled = smoothing !== false;
    if (smoothing) ctx.imageSmoothingQuality = smoothing;
    if (fill === "fill") {
        ctx.scale(Math.min(w / data.width, 1), Math.min(h / data.height, 1));
    } else {
        ctx.scale(w / data.width, h / data.height);
    }
    ctx.drawImage(x, 0, 0);
    return ctx;
}
export function data2canvas(data: ImageData, w?: number, h?: number) {
    const x = newCanvas(w || data.width, h || data.height);
    const ctx = x.getContext("2d")!;
    ctx.putImageData(data, 0, 0);
    return x;
}
/**
 * 转成 PaddleOCR 输入。注意：PaddleOCR 模型（det/rec/cls）均按 BGR 通道顺序导出
 * （img_mode: BGR），所以返回值顺序是 [B, G, R]，mean/std 按输出通道顺序应用。
 */
export function toPaddleInput(image: ImageData, mean: number[], std: number[]) {
    const imagedata = image.data;
    const redArray: number[][] = [];
    const greenArray: number[][] = [];
    const blueArray: number[][] = [];
    let x = 0;
    let y = 0;
    for (let i = 0; i < imagedata.length; i += 4) {
        if (!blueArray[y]) blueArray[y] = [];
        if (!greenArray[y]) greenArray[y] = [];
        if (!redArray[y]) redArray[y] = [];
        // imagedata 是 RGBA；输出 [B, G, R]（BGR），mean/std 依次对应 B, G, R
        blueArray[y][x] = (imagedata[i + 2] / 255 - mean[0]) / std[0];
        greenArray[y][x] = (imagedata[i + 1] / 255 - mean[1]) / std[1];
        redArray[y][x] = (imagedata[i] / 255 - mean[2]) / std[2];
        x++;
        if (x === image.width) {
            x = 0;
            y++;
        }
    }

    return [blueArray, greenArray, redArray];
}
/**
 * 用 DLT 求解单应矩阵 H（h22 = 1），使得每组对应点 (x,y)->(u,v) 满足：
 *   u = (h0*x + h1*y + h2) / (h6*x + h7*y + 1)
 *   v = (h3*x + h4*y + h5) / (h6*x + h7*y + 1)
 * 奇异时返回 null。
 */
export function solveHomography(src: [number, number][], dst: [number, number][]): number[] | null {
    // 8x8 线性方程组
    const a: number[][] = [];
    const b: number[] = [];
    for (let i = 0; i < 4; i++) {
        const [x, y] = src[i];
        const [u, v] = dst[i];
        a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
        b.push(u);
        a.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
        b.push(v);
    }
    // 高斯消元（列主元）
    const m = a.map((row, r) => [...row, b[r]]);
    for (let col = 0; col < 8; col++) {
        let piv = col;
        for (let r = col + 1; r < 8; r++) {
            if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
        }
        if (Math.abs(m[piv][col]) < 1e-12) return null;
        [m[col], m[piv]] = [m[piv], m[col]];
        for (let r = 0; r < 8; r++) {
            if (r === col) continue;
            const f = m[r][col] / m[col][col];
            for (let c = col; c <= 8; c++) m[r][c] -= f * m[col][c];
        }
    }
    return m.map((row, r) => row[8] / m[r][r]);
}

/**
 * 透视变换：把源图中四边形 quad 映射到 (dstW, dstH) 的矩形（比仿射更精确地矫正透视）。
 * 逐像素用逆映射做双线性采样，越界取最近边缘像素（等效 BORDER_REPLICATE）。
 * 单应矩阵求解失败时返回 null，调用方回退到仿射路径。
 */
export function warpPerspective(
    src: ImageData,
    dstW: number,
    dstH: number,
    quad: [number, number][],
): { data: Uint8ClampedArray; width: number; height: number } | null {
    // 目标矩形四角 -> 源四边形（逆映射，便于逐像素反查源图）
    const h = solveHomography(
        [
            [0, 0],
            [dstW, 0],
            [dstW, dstH],
            [0, dstH],
        ],
        quad,
    );
    if (!h) return null;

    const [h0, h1, h2, h3, h4, h5, h6, h7] = h;
    const sw = src.width;
    const sh = src.height;
    const sData = src.data;
    const out = new Uint8ClampedArray(dstW * dstH * 4);

    for (let y = 0; y < dstH; y++) {
        for (let x = 0; x < dstW; x++) {
            const denom = h6 * x + h7 * y + 1;
            const sx = (h0 * x + h1 * y + h2) / denom;
            const sy = (h3 * x + h4 * y + h5) / denom;
            const x0 = Math.floor(sx);
            const y0 = Math.floor(sy);
            const fx = sx - x0;
            const fy = sy - y0;
            const cx0 = clip(x0, 0, sw - 1);
            const cy0 = clip(y0, 0, sh - 1);
            const cx1 = clip(x0 + 1, 0, sw - 1);
            const cy1 = clip(y0 + 1, 0, sh - 1);
            const i00 = (cy0 * sw + cx0) * 4;
            const i10 = (cy0 * sw + cx1) * 4;
            const i01 = (cy1 * sw + cx0) * 4;
            const i11 = (cy1 * sw + cx1) * 4;
            const o = (y * dstW + x) * 4;
            for (let c = 0; c < 3; c++) {
                const v00 = sData[i00 + c];
                const v10 = sData[i10 + c];
                const v01 = sData[i01 + c];
                const v11 = sData[i11 + c];
                const top = v00 + (v10 - v00) * fx;
                const bot = v01 + (v11 - v01) * fx;
                out[o + c] = top + (bot - top) * fy;
            }
            out[o + 3] = 255;
        }
    }
    return { data: out, width: dstW, height: dstH };
}

export type AsyncType<T> = T extends Promise<infer U> ? U : never;
export type SessionType = AsyncType<ReturnType<typeof import("onnxruntime-common").InferenceSession.create>>;

export class tLog {
    private tl: { t: string; n: number }[] = [];
    private name: string;
    constructor(taskName: string) {
        this.name = taskName;
    }
    l(name: string) {
        const now = performance.now();
        this.tl.push({ t: name, n: now });
        const l: { d: number; n: string; c: number }[] = [];
        for (let i = 1; i < this.tl.length; i++) {
            const d = this.tl[i].n - this.tl[i - 1].n;
            const name = this.tl[i - 1].t;
            const f = l.find((x) => x.n === name);
            if (f) {
                f.c++;
                f.d += d;
            } else l.push({ d: d, n: name, c: 1 });
        }
        const x: string[] = [];
        for (const i of l) {
            const t = i.c > 1 ? `${i.n}x${i.c}` : i.n;
            x.push(`${t} ${i.d}`);
        }
        x.push((this.tl.at(-1) as (typeof this.tl)[0]).t);
        console.log(`${this.name} ${l.map((i) => i.d).reduce((p, c) => p + c, 0)}ms: `, x.join(" "));
    }
}
