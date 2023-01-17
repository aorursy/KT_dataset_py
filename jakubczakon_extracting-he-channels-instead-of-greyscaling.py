%matplotlib inline
%load_ext autoreload
%autoreload 2

import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.color import rgb2grey, rgb2hed
from skimage.exposure import rescale_intensity
from sklearn.externals import joblib


def plot_list(images=[], labels=[], n_rows=1):
    n_img = len(images)
    n_lab = len(labels)
    n_cols = math.ceil((n_lab+n_img)/n_rows)
    plt.figure(figsize=(12,10))
    for i, image in enumerate(images):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(image)
    for j, label in enumerate(labels):
        plt.subplot(n_rows,n_cols,n_img+j+1)
        plt.imshow(label, cmap='nipy_spectral')
    plt.show()
sample_images = joblib.load('../input/sample_stained_not_stained_images.pkl')
plot_list(sample_images,n_rows=4)
def is_stained(img):
    red_mean, green_mean, blue_mean = img.mean(axis=(0, 1))
    if red_mean == green_mean == blue_mean:
        return False
    else:
        return True
for img in sample_images:
    if is_stained(img):
        img_hed = rgb2hed(img)
        img_hematoxilin = img_hed[:,:,0]
        img_eosin = img_hed[:,:,1]
        img_dab = img_hed[:,:,2]

        plot_list([img, img_hematoxilin, img_eosin, img_dab])
def stain_deconvolve(img, mode='hematoxylin_eosin_sum'):
    img_hed = rgb2hed(img)
    if mode == 'hematoxylin_eosin_sum':
        h, w = img.shape[:2]
        img_hed = rgb2hed(img)
        img_he_sum = np.zeros((h, w, 2))
        img_he_sum[:, :, 0] = rescale_intensity(img_hed[:, :, 0], out_range=(0, 1))
        img_he_sum[:, :, 1] = rescale_intensity(img_hed[:, :, 1], out_range=(0, 1))
        img_deconv = rescale_intensity(img_he_sum.sum(axis=2), out_range=(0, 1))
    elif mode == 'hematoxylin':
        img_deconv = img_hed[:, :, 0]
    elif mode == 'eosin':
        img_deconv = img_hed[:, :, 1]
    else:
        raise NotImplementedError('only hematoxylin_eosin_sum, hematoxylin, eosin modes are supported')
    return img_deconv
for img in sample_images:
    if is_stained(img):
        deconv = stain_deconvolve(img)
        grey = 1-rgb2grey(img)
        plot_list([img, grey, deconv])
