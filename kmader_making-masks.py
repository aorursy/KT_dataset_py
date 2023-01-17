from skimage.io import imread # for reading images

import matplotlib.pyplot as plt # for showing plots

from skimage.measure import label # for labeling regions

import numpy as np # for matrix operations and array support

from skimage.color import label2rgb # for making overlay plots

import matplotlib.patches as mpatches # for showing rectangles and annotations

from skimage.morphology import medial_axis # for finding the medial axis and making skeletons

from skimage.morphology import skeletonize, skeletonize_3d # for just the skeleton code

import pandas as pd # for reading the swc files (tables of somesort)

import os

from glob import glob # for lists of files

def read_swc(in_path):

    swc_df = pd.read_csv(in_path, sep = ' ', comment='#', 

                         header = None)

    # a pure guess here

    swc_df.columns = ['id', 'junk1', 'x', 'y', 'junk2', 'width', 'next_idx']

    return swc_df[['x', 'y', 'width']]

DATA_ROOT = '../input/road'
image_files = glob(os.path.join(DATA_ROOT, '*.tif'))

image_mask_files = [(c_file, '.swc'.join(c_file.split('.tif'))) for c_file in image_files]

print(image_mask_files)
%matplotlib inline

im_path, mask_path = image_mask_files[0]

im_data = imread(im_path)

mk_data = read_swc(mask_path)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 3))

ax1.imshow(im_data)

ax2.plot(mk_data['x'], mk_data['y'], 'b.')

ax3.imshow(im_data)

ax3.plot(mk_data['x'], mk_data['y'], 'r.')
fig, m_axes = plt.subplots(5,3, figsize = (12, 20))

for (im_path, mask_path), c_ax in zip(image_mask_files, m_axes.flatten()):

    im_data = imread(im_path)

    mk_data = read_swc(mask_path)

    c_ax.imshow(im_data)

    c_ax.plot(mk_data['x'], mk_data['y'], 'r.')

fig.savefig('all_scenes.pdf')