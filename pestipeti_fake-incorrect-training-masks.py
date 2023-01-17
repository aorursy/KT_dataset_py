import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio

from scipy import ndimage
from pathlib import Path

im_dir = Path('../input/train/')

df_pred = pd.read_csv('../input/train.csv', index_col=[0])
df_pred.fillna('', inplace=True)
df_pred['suspicious'] = False

for index, row in df_pred.iterrows():
    encoded_mask = row['rle_mask'].split(' ')
    if len(encoded_mask) > 1 and len(encoded_mask) < 5 and int(encoded_mask[1]) % 101 == 0:
        df_pred.loc[index,'suspicious'] = True

def show_plot(rows):
    idx_images = 0
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)

    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

    for index, row in rows.iterrows():
        im_path = im_dir / 'images' / '{}.png'.format(index)
        img = imageio.imread(im_path.as_posix())

        im_path = im_dir / 'masks' / '{}.png'.format(index)
        mask = imageio.imread(im_path.as_posix())

        ax = axs[int(idx_images / grid_width), idx_images % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.45, cmap="Greens")

        ax.text(1, 1, np.count_nonzero(mask), color="black", ha="left", va="top")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        idx_images = idx_images + 1

        if idx_images > 59:
            break

    plt.suptitle("Grey: training image, Green: training mask, Top-left: # of pixels")
show_plot(df_pred[df_pred['suspicious'] == False])
show_plot(df_pred[df_pred['suspicious'] == True])
ok = len(df_pred[df_pred['suspicious'] == False])
fake = len(df_pred[df_pred['suspicious'] == True])

print("There are {} suspicious masks ({:.2f}%) in the training set!".format(fake, fake/(ok + fake) * 100))