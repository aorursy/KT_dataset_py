import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from skimage.io import imread

from skimage.transform import downscale_local_mean

from os.path import join

from tqdm import tqdm



input_folder = join('..', 'input')



df_mask = pd.read_csv(join(input_folder, 'train_masks.csv'), usecols=['img'])

ids_train = df_mask['img'].map(lambda s: s.split('_')[0]).unique()



imgs_idx = list(range(1, 17))
load_img = lambda im, idx: imread(join(input_folder, 'train', '{}_{:02d}.jpg'.format(im, idx)))

load_mask = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))

resize = lambda im: downscale_local_mean(im, (4,4) if im.ndim==2 else (4,4,1)).astype(np.float32) / 255

mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))
im = resize(load_img(ids_train[2], 7))

im_mask = resize(load_mask(ids_train[2], 7))



fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(im)

ax[1].imshow(im_mask)

ax[2].imshow(mask_image(im, im_mask))
mean_mask_idx = {}

for cnt, img_id in enumerate(ids_train):

    for img_idx in imgs_idx:

        im_mask = resize(load_mask(img_id, img_idx))

        if img_idx in mean_mask_idx:

            mean_mask_idx[img_idx] = (mean_mask_idx[img_idx] * cnt + im_mask) / (cnt+1)

        else:

            mean_mask_idx[img_idx] = im_mask.astype(np.float64)
fig, ax = plt.subplots(8, 2, figsize=(8, 32))

ax = ax.ravel()

for i, img_idx in enumerate(imgs_idx):

    ax[i].imshow(mean_mask_idx[img_idx])

    ax[i].set_title(str(img_idx))