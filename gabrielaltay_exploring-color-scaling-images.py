import tifffile as tiff

import numpy as np

import matplotlib.pyplot as plt

import skimage.exposure
fname = '../input/three_band/6120_2_4.tif'

tfl = tiff.TiffFile(fname)

pg = tfl.pages[0]

print('bits per sample: {}'.format(pg.bits_per_sample))

img = tfl.asarray()

print('img dtype: {}'.format(img.dtype))

print('img shape: {}'.format(img.shape))

print('img min/max pixel values: {}, {}'.format(img.min(), img.max()))



# most python things expect the color channels as the last dimension and floats

new_im = np.zeros(shape=(img.shape[1], img.shape[2], img.shape[0]), dtype=img.dtype)

for ich in range(img.shape[0]):

    new_im[:,:,ich] = img[ich,:,:]

img = new_im.astype(np.float32)

img = skimage.exposure.rescale_intensity(img)  # will scale float values between 0.0 and 1.0

print()

print('img dtype: {}'.format(img.dtype))

print('img shape: {}'.format(img.shape))

print('img min/max pixel values: {}, {}'.format(img.min(), img.max()))

img_orig = img.copy()
# plot RGB

img = img_orig.copy()

fig, axes = plt.subplots(figsize=(6,12), nrows=2, ncols=1)

axes[0].imshow(img)

for ich, color, ls in zip(range(3), ['red', 'green', 'blue'], ['-', '--', ':']):

    n, bins, patches = axes[1].hist(

        img[:,:,ich].flatten(), bins=20, color=color, histtype='step', lw=3.0, ls=ls)
# what happens if we linearly stretch values to fill the range 0-1

# with a percentile clip to remove extreme values

img = img_orig.copy()

vmin = np.percentile(img, 2.0)

vmax = np.percentile(img, 98.0)

img[img<vmin] = vmin

img[img>vmax] = vmax

img = (img - img.min()) / (img.max() - img.min())

    

# plot RGB

fig, axes = plt.subplots(figsize=(6,12), nrows=2, ncols=1)

axes[0].imshow(img)

for ich, color, ls in zip(range(3), ['red', 'green', 'blue'], ['-', '--', ':']):

    n, bins, patches = axes[1].hist(

        img[:,:,ich].flatten(), bins=20, color=color, histtype='step', lw=3.0, ls=ls)
# what happens if we linearly stretch each color channel to fill the range 0-1

# this seems to be what the plotRGB function in R does with stretch='lin'

img = img_orig.copy()

for ich in range(img.shape[2]):

    im = img[:,:,ich]

    vmin = np.percentile(im, 2.0)

    vmax = np.percentile(im, 98.0)

    im[im<vmin] = vmin

    im[im>vmax] = vmax

    im = (im - im.min()) / (im.max() - im.min())

    img[:,:,ich] = im

    

# plot RGB

fig, axes = plt.subplots(figsize=(6,12), nrows=2, ncols=1)

axes[0].imshow(img)

for ich, color, ls in zip(range(3), ['red', 'green', 'blue'], ['-', '--', ':']):

    n, bins, patches = axes[1].hist(

        img[:,:,ich].flatten(), bins=20, color=color, histtype='step', lw=3.0, ls=ls)
# what happens if we equalize histogram each color channel 

img = img_orig.copy()

for ich in range(img.shape[2]):

    im = img[:,:,ich]

    im = skimage.exposure.equalize_hist(im)

    img[:,:,ich] = im



# plot RGB

fig, axes = plt.subplots(figsize=(6,12), nrows=2, ncols=1)

axes[0].imshow(img)

for ich, color, ls in zip(range(3), ['red', 'green', 'blue'], ['-', '--', ':']):

    n, bins, patches = axes[1].hist(

        img[:,:,ich].flatten(), bins=20, color=color, histtype='step', lw=3.0, ls=ls)    

    
# what happens if we equalize adaptive histogram each color channel 

img = img_orig.copy()

for ich in range(img.shape[2]):

    im = img[:,:,ich]

    im = skimage.exposure.equalize_adapthist(im)

    img[:,:,ich] = im



# plot RGB

fig, axes = plt.subplots(figsize=(6,12), nrows=2, ncols=1)

axes[0].imshow(img)

for ich, color, ls in zip(range(3), ['red', 'green', 'blue'], ['-', '--', ':']):

    n, bins, patches = axes[1].hist(

        img[:,:,ich].flatten(), bins=20, color=color, histtype='step', lw=3.0, ls=ls)