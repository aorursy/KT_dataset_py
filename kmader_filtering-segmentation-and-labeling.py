from skimage.io import imread # for reading images

import matplotlib.pyplot as plt # for showing plots

from skimage.filters import median # for filtering the data

from skimage.measure import label # for labeling bubbles

from skimage.morphology import disk # for morphology neighborhoods

from skimage.morphology import erosion, dilation, opening # for disconnecting bubbles

import numpy as np # for matrix operations and array support
foam_crop = imread('../input/rec_8bit_ph03_cropC.tif')

print("Data Loaded, Dimensions", foam_crop.shape)

foam_slice = np.random.permutation(foam_crop)[0]

print("Slice Loaded, Dimensions", foam_slice.shape)
# show the slice and histogram

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))

ax1.imshow(foam_slice, cmap = 'gray')

ax1.axis('off')

ax2.hist(foam_slice.ravel()) # make it 1d to make a histogram

ax2.set_title('Intensity Histogram')
# try a filter and a basic threshold (values lower than 110)

sub_img = foam_slice

filt_img = median(sub_img, disk(4))

thresh_img = filt_img<55

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

ax1.imshow(sub_img, cmap = 'gray')

ax2.imshow(filt_img, cmap = 'gray')

ax3.imshow(thresh_img, cmap = 'gray')
# erode the bubbles apart

eroded_img = erosion(thresh_img.copy(), disk(4))

# iteratively erode and open

for i in range(14): # alternate eroding and 

    eroded_img = erosion(eroded_img, disk(1))

    eroded_img = opening(eroded_img, disk(8))

bubble_img = eroded_img

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

ax1.imshow(sub_img, cmap = 'gray')

ax2.imshow(thresh_img, cmap = 'gray')

ax3.imshow(bubble_img, cmap = 'gray')
# make connected component labels

label_image = label(bubble_img)

# reorder the labels to make it easier to see

new_label_image = np.zeros_like(label_image)

label_idxs = [i for i in np.random.permutation(np.unique(label_image)) if i>0]

for new_label, old_label in enumerate(label_idxs):

    new_label_image[label_image==old_label] = new_label

# show the image

fig, (ax1, ax2) = plt.subplots(1,2 , figsize = (12,  4))

ax1.imshow(sub_img, cmap = 'gray')

ax2.imshow(new_label_image, cmap = plt.cm.gist_earth)