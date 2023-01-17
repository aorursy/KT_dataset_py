import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
%matplotlib inline

import numpy as np

import pandas as pd

from skimage.util import montage as montage2d

from skimage.color import rgb2hsv, gray2rgb

from skimage.io import imread

import os

from glob import glob

base_img_dir = os.path.join('..', 'input')
all_tails = glob(os.path.join(base_img_dir, '*', '*', '*.jpg'))

print(len(all_tails), 'tails found')
# load images

sample_tails = [imread(c_path) for c_path in np.random.choice(all_tails, size=25)] 

# make sure all are color

sample_rgb_tails = [c_img if len(np.shape(c_img))==3 else gray2rgb(c_img) for c_img in sample_tails]

fig, m_axs = plt.subplots(5, 5, figsize=(20, 20))

for c_ax, c_img in zip(m_axs.flatten(), sample_rgb_tails):

    c_ax.imshow(c_img)

    c_ax.axis('off')
fig, m_axs = plt.subplots(5, 5, figsize=(20, 20))

for c_ax, c_img in zip(m_axs.flatten(), sample_rgb_tails):

    for i, c_col in enumerate(['red', 'green', 'blue']):

        c_ax.hist(c_img[:, :, i].ravel(), np.linspace(0, 255, 30), alpha=0.5, label=c_col)

    c_ax.legend()

    c_ax.axis('off')
fig, m_axs = plt.subplots(5, 5, figsize=(20, 20))

for c_ax, c_img in zip(m_axs.flatten(), sample_rgb_tails):

    hsv_img = rgb2hsv(c_img)

    for i, c_col in enumerate(['hue', 'saturation', 'value']):

        c_ax.hist(hsv_img[:, :, i].ravel(), np.linspace(0, 1, 30), alpha=0.5, label=c_col)

    c_ax.legend()

    c_ax.axis('off')
fig, m_axs = plt.subplots(5, 5, figsize=(20, 20))

for c_ax, c_img in zip(m_axs.flatten(), sample_rgb_tails):

    c_ax.imshow(montage2d(rgb2hsv(c_img).swapaxes(0, 2).swapaxes(1, 2)))

    c_ax.axis('off')
from skimage.feature import canny

canny_value_tails = [canny(rgb2hsv(c_img)[:, :, 2], sigma=3) for c_img in sample_rgb_tails]
fig, m_axs = plt.subplots(5, 5, figsize=(20, 20))

for c_ax, c_img in zip(m_axs.flatten(), canny_value_tails):

    c_ax.imshow(c_img)

    c_ax.axis('off')
from skimage.morphology import label

def label_size(in_img, percentile=0):

    s_label = label(in_img)

    new_img = np.zeros_like(s_label)

    for i in np.unique(s_label[s_label>0]):

        new_img[s_label==i] = np.sum(s_label==i)

    # keep only big enough components

    return new_img>np.percentile(new_img[new_img>0], percentile)

clean_canny_tails = [label_size(c_img, 70) for c_img in canny_value_tails]
fig, m_axs = plt.subplots(5, 5, figsize=(20, 20))

for c_ax, c_img in zip(m_axs.flatten(), clean_canny_tails):

    c_ax.imshow(c_img)

    c_ax.axis('off')
from scipy.ndimage import distance_transform_edt

def padded_inverse_dt(in_img, padding=None):

    if padding is None:

        in_shape = np.shape(in_img)

        padding = [int(0.75*in_shape[0]), int(0.75*in_shape[1])]

    base_img = np.pad(in_img, 

                      [(padding[0], padding[0]), (padding[1], padding[1])], 

                      mode='constant', 

                      constant_values=0)

    dist_img = distance_transform_edt(1-base_img)

    return dist_img[padding[0]:-padding[0], padding[1]:-padding[1]]

dt_canny_tails = [padded_inverse_dt(c_img) for c_img in canny_value_tails]
fig, m_axs = plt.subplots(5, 5, figsize=(20, 20))

for c_ax, c_img in zip(m_axs.flatten(), dt_canny_tails):

    c_ax.imshow(c_img)

    c_ax.axis('off')