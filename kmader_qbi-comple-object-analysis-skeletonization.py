from skimage.io import imread # for reading images

import matplotlib.pyplot as plt # for showing plots

from skimage.measure import label # for labeling regions

import numpy as np # for matrix operations and array support

from skimage.color import label2rgb # for making overlay plots

import matplotlib.patches as mpatches # for showing rectangles and annotations

from skimage.morphology import medial_axis # for finding the medial axis and making skeletons

from skimage.morphology import skeletonize, skeletonize_3d # for just the skeleton code

from skimage.measure import regionprops
em_image_vol = imread('../input/training.tif')

em_thresh_vol = imread('../input/training_groundtruth.tif')

em_label_vol = label(em_thresh_vol)

print("Data Loaded, Dimensions", em_image_vol.shape,'->',em_thresh_vol.shape, 'Max Labels:', em_label_vol.shape)
em_idx = 100

em_slice = em_image_vol[:,em_idx]

# take the maximum threshold or label so we get more complete shapes for the demo

em_thresh = np.max(em_thresh_vol,1)#[em_idx]

em_label = np.max(em_label_vol,1)#[:,em_idx]

print("Slice Loaded, Dimensions", em_slice.shape)
# show the slice and threshold

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (9, 6))

ax1.imshow(em_slice, cmap = 'gray')

ax1.axis('off')

ax1.set_title('Image')

ax2.imshow(em_label, cmap = 'gist_earth')

ax2.axis('off')

ax2.set_title('Segmentation')

# here we mark the threshold on the original image



ax3.imshow(label2rgb(em_label,em_slice, bg_label=0))

ax3.axis('off')

ax3.set_title('Overlayed')
# Borrowed from (http://scikit-image.org/docs/0.10.x/auto_examples/plot_medial_transform.html)

# Compute the medial axis (skeleton) and the distance transform

skel, distance = medial_axis(em_thresh, return_distance=True)

skel_morph = skeletonize(em_thresh>0)



# Distance to the background for pixels of the skeleton

dist_on_skel = distance * skel



fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))

ax1.imshow(distance, cmap='nipy_spectral', interpolation='nearest')

ax1.contour(em_thresh, [0.5], colors='w')

ax1.axis('off')

ax1.set_title('Distance Map')



ax2.imshow(skel_morph, cmap = 'gray')

ax2.set_title('Skeleton')

ax2.axis('off')

ax3.imshow(dist_on_skel, cmap=plt.cm.nipy_spectral, interpolation='nearest')

ax3.contour(em_thresh, [0.5], colors='w')

ax3.axis('off')



fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
%%time

# Borrowed from (http://scikit-image.org/docs/0.10.x/auto_examples/plot_medial_transform.html)

# Compute the medial axis (skeleton) and the distance transform

skel = skeletonize_3d(em_thresh_vol>0)
%%time

from scipy.ndimage.morphology import distance_transform_edt as distmap

dist_3d = distmap(em_thresh_vol)
dist_on_skel=dist_3d*skel

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))

ax1.imshow(dist_3d[:,em_idx], cmap='nipy_spectral', interpolation='nearest')

ax1.contour(em_thresh_vol[:,em_idx], [0.5], colors='w')

ax1.axis('off')

ax1.set_title('Distance Map')

ax2.imshow(np.max(skel,1), cmap = 'gray')

ax2.set_title('Skeleton')

ax2.axis('off')

ax3.imshow(np.max(dist_on_skel,1), cmap=plt.cm.nipy_spectral, interpolation='nearest')

ax3.contour(np.sum(em_thresh_vol,1), [0.5], colors='w')

ax3.axis('off')



fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 11))

ax1.imshow(np.max(dist_3d,0), cmap='nipy_spectral', interpolation='nearest')

ax1.contour(np.sum(em_thresh_vol,0), [0.5], colors='w')

ax1.axis('off')

ax1.set_title('Distance Map')

ax2.imshow(np.max(skel,0), cmap = 'gray')

ax2.set_title('Skeleton')

ax2.axis('off')

ax3.imshow(np.max(dist_on_skel,0), cmap=plt.cm.nipy_spectral, interpolation='nearest')

ax3.contour(np.sum(em_thresh_vol,0), [0.5], colors='w')

ax3.set_title('Skeleton Distance')

ax3.axis('off')



ax4.imshow(np.max(skel,0)*(np.argmax(skel,0)-skel.shape[0]/2.0), 

           cmap=plt.cm.seismic, interpolation='nearest')

ax4.contour(np.sum(em_thresh_vol,0), [0.5], colors='k')

ax4.set_title('Skeleton Depth')

ax4.axis('off')



fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)

fig.savefig('nice_skeleton_map.pdf')
shape_analysis_list = regionprops(em_label)

first_region = shape_analysis_list[0]

print('List of region properties for',len(shape_analysis_list), 'regions')

print('Features Calculated:',', '.join([f for f in dir(first_region) if not f.startswith('_')]))