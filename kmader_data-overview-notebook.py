from skimage.io import imread # for reading images

import matplotlib.pyplot as plt # for showing plots

from skimage.filters import median # for filtering the data

from skimage.measure import label # for labeling bubbles

from skimage.morphology import disk # for morphology neighborhoods

from skimage.morphology import erosion, dilation, opening # for disconnecting bubbles

import numpy as np # for matrix operations and array support
em_image = imread('../input/training.tif')

em_labels = imread('../input/training_groundtruth.tif')

print("Data Loaded, Dimensions", em_image.shape,'->',em_labels.shape)
em_idx = np.random.permutation(range(em_image.shape[0]))[0]

em_slice = em_image[em_idx]

print("Slice Loaded, Dimensions", em_slice.shape)

# show the slice and histogram

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))

ax1.imshow(em_slice, cmap = 'gray')

ax1.axis('off')

ax2.hist(em_slice.ravel()) # make it 1d to make a histogram

ax2.set_title('Intensity Histogram')
# try a filter and a basic threshold (values lower than 110)

sub_img = em_slice

filt_img = median(sub_img, disk(4))

thresh_img = filt_img<130

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

ax1.imshow(sub_img, cmap = 'gray')

ax2.imshow(filt_img, cmap = 'gray')

ax3.imshow(thresh_img, cmap = 'gray')
# erode the bubbles apart

eroded_img = erosion(thresh_img.copy(), disk(6))

# iteratively erode and open

for i in range(10): # alternate eroding and opening

    eroded_img = opening(eroded_img, disk(2))

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
# make connected component labels

real_label_image = label(em_labels[em_idx])

# reorder the labels to make it easier to see

new_label_image = np.zeros_like(real_label_image)

label_idxs = [i for i in np.random.permutation(np.unique(real_label_image)) if i>0]

for new_label, old_label in enumerate(label_idxs):

    new_label_image[real_label_image==old_label] = new_label

# show the image

fig, (ax1, ax2) = plt.subplots(1,2 , figsize = (12,  4))

ax1.imshow(sub_img, cmap = 'gray')

ax2.imshow(new_label_image, cmap = plt.cm.gist_earth)