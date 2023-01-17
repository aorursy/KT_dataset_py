import os

from functools import reduce



import pandas as pd

import skimage.io

import numpy as np

import matplotlib.pyplot as plt
INPUT_DIR = "../input/prostate-cancer-grade-assessment"

TRAIN_DIR = f"{INPUT_DIR}/train_images"

MASK_DIR = f"{INPUT_DIR}/train_label_masks"



BLACK = (0,) * 3

GRAY = (200,) * 3

WHITE = (255,) * 3

RED = (255, 0, 0)



SIZE = 128

N = 16
train = pd.read_csv(f"{INPUT_DIR}/train.csv")

train.head()
def imread(path):

    if not os.path.exists(path):

        raise FileNotFoundError(f"No such file or directory: '{path}'")



    return skimage.io.MultiImage(path)





def imshow(

    img,

    title=None,

    show_shape=True,

    figsize=(8, 8)

):

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(img)

    ax.grid("off")

    ax.set_xticks([])

    ax.set_yticks([])



    if show_shape:

        ax.set_xlabel(f"Shape: {img.shape}", fontsize=16)

        

    if title:

        ax.set_title(title, fontsize=16)



    return ax
img_id = "000920ad0b612851f8e01bcc880d9b3d"

img_org = imread(os.path.join(TRAIN_DIR, f"{img_id}.tiff"))[-1]

imshow(img_org, "Original image")
H, W = img_org.shape[:2]

pad_h = (SIZE - H % SIZE) % SIZE

pad_w = (SIZE - W % SIZE) % SIZE



print("pad_h:", pad_h)

print("pad_w", pad_w)
padded_vis = np.pad(

    img_org,

    [[pad_h // 2, pad_h - pad_h // 2],

     [pad_w // 2, pad_w - pad_w // 2],

     [0, 0]],

    constant_values=GRAY[0],  # use GRAY for visualization.

)



imshow(padded_vis, "Padded image")
padded = np.pad(

    img_org,

    [[pad_h // 2, pad_h - pad_h // 2],

     [pad_w // 2, pad_w - pad_w // 2],

     [0, 0]],

    constant_values=WHITE[0],

)
N_ROWS = padded.shape[0] // SIZE

N_COLS = padded.shape[1] // SIZE



print("N_ROWS :", N_ROWS)

print("N_COLS :", N_COLS)

print("N_TILES:", N_ROWS * N_COLS)
reshaped = padded.reshape(

    padded.shape[0] // SIZE,

    SIZE,

    padded.shape[1] // SIZE,

    SIZE,

    3,

)

transposed = reshaped.transpose(0, 2, 1, 3, 4)

tiles = transposed.reshape(-1, SIZE, SIZE, 3)



print("reshaped.shape  :", reshaped.shape)

print("transposed.shape:", transposed.shape)

print("tiles.shape     :", tiles.shape)
def merge_tiles(tiles, funcs=None):

    """

    If `funcs` specified, apply them to each tile before merging.

    """

    return np.vstack([

        np.hstack([

            reduce(lambda acc, f: f(acc), funcs, x) if funcs else x

            for x in row

        ])

        for row in tiles

    ])





def draw_borders(img):

    """

    Put borders around an image.

    """

    ret = img.copy()

    ret[0, :] = GRAY   # top

    ret[-1, :] = GRAY  # bottom

    ret[:, 0] = GRAY   # left

    ret[:, -1] = GRAY  # right

    return ret

imshow(merge_tiles(transposed, [draw_borders]), "Tiles")
sums = tiles.reshape(tiles.shape[0], -1).sum(axis=-1)



highlight = lambda x: "color: {}".format("red" if x != sums.max() else "black")

pd.DataFrame(sums.reshape(N_ROWS, N_COLS)).style.applymap(highlight)
idxs_selected = np.argsort(sums)[:N]

idxs_selected
def fill_tiles(tiles, fill_func):

    """

    Fill each tile with another array created by `fill_func`.

    """

    return np.array([[fill_func(x) for x in row] for row in tiles])





def make_patch_func(true_color, false_color):

    def ret(x):

        """

        Retunrs a color patch. The color will be `true_color` if `x` is True otherwise `false_color`.

        """

        color = true_color if x else false_color

        return np.tile(color, (SIZE, SIZE, 1)).astype(np.uint8)



    return ret
mask = np.isin(np.arange(len(sums)), idxs_selected).reshape(N_ROWS, N_COLS)

mask = fill_tiles(mask, make_patch_func(WHITE, BLACK))

mask = merge_tiles(mask, [draw_borders])



imshow(mask, "Selected tiles")
mask = np.isin(np.arange(len(sums)), idxs_selected).reshape(N_ROWS, N_COLS)

mask = fill_tiles(mask, make_patch_func(RED, WHITE))

mask = merge_tiles(mask, [draw_borders])



with_mask = np.ubyte(0.7 * padded + 0.3 * mask)



imshow(with_mask, "Selected tiles")