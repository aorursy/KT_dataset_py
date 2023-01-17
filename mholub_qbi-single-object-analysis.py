from skimage.io import imread # for reading images

import matplotlib.pyplot as plt # for showing plots

from skimage.measure import label # for labeling regions

from skimage.measure import regionprops # for shape analysis

import numpy as np # for matrix operations and array support

from skimage.color import label2rgb # for making overlay plots

import matplotlib.patches as mpatches # for showing rectangles and annotations
# simple test image diagonal

test_img=np.eye(4)

print('Input Image')

print(test_img)



test_label_4=label(test_img,connectivity=1)

print('Labels with 4-neighborhood')

print(test_label_4)



test_label_8=label(test_img,connectivity=2)

print('Labels with 8-neighborhood')

print(test_label_8)
test_img=np.array([1 if x in [0,13,26] else 0 for x in range(27)]).reshape((3,3,3))

print('Input Image')

print(test_img)



test_label_1=label(test_img,connectivity=1)

print('Labels with Face-sharing')

print(test_label_1)



test_label_2=label(test_img,connectivity=2)

print('Labels with Edge-Sharing')

print(test_label_2)



test_label_3=label(test_img,connectivity=3)

print('Labels with Vertex-Sharing')

print(test_label_3)
em_image_vol = imread('../input/training.tif')

em_thresh_vol = imread('../input/training_groundtruth.tif')

print("Data Loaded, Dimensions", em_image_vol.shape,'->',em_thresh_vol.shape)
em_idx = np.random.permutation(range(em_image_vol.shape[0]))[0]

em_slice = em_image_vol[em_idx]

em_thresh = em_thresh_vol[em_idx]

print("Slice Loaded, Dimensions", em_slice.shape)
# show the slice and threshold

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (9, 4))

ax1.imshow(em_slice, cmap = 'gray')

ax1.axis('off')

ax1.set_title('Image')

ax2.imshow(em_thresh, cmap = 'gray')

ax2.axis('off')

ax2.set_title('Segmentation')

# here we mark the threshold on the original image



ax3.imshow(label2rgb(em_thresh,em_slice, bg_label=0))

ax3.axis('off')

ax3.set_title('Overlayed')
# make connected component labels

em_label = label(em_thresh)



# show the segmentation, labels and overlay

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (9, 4))

ax1.imshow(em_thresh, cmap = 'gray')

ax1.axis('off')

ax1.set_title('Segmentation')

ax2.imshow(em_label, cmap = plt.cm.gist_earth)

ax2.axis('off')

ax2.set_title('Labeling')

# here we mark the threshold on the original image



ax3.imshow(label2rgb(em_label,em_slice, bg_label=0))

ax3.axis('off')

ax3.set_title('Overlayed')
shape_analysis_list = regionprops(em_label)

first_region = shape_analysis_list[0]

print('List of region properties for',len(shape_analysis_list), 'regions')

print('Features Calculated:',', '.join([f for f in dir(first_region) if not f.startswith('_')]))
fig, ax = plt.subplots(figsize=(10, 6))

ax.imshow(label2rgb(em_label,em_slice, bg_label=0))



for region in shape_analysis_list:

    # draw rectangle using the bounding box

    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                              fill=False, edgecolor='red', linewidth=2)

    ax.add_patch(rect)



ax.set_axis_off()

plt.tight_layout()
fig, ax = plt.subplots(figsize=(10, 6))

ax.imshow(label2rgb(em_label,em_slice, bg_label=0))



for region in shape_analysis_list:

    x1=region.major_axis_length

    x2=region.minor_axis_length

    anisotropy = (x1-x2)/((x1))

    # for anisotropic shapes use red for the others use blue

    print('Label:',region.label,'Anisotropy %2.2f' % anisotropy)

    if anisotropy>0.1:

        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                              fill=False, edgecolor='red', linewidth=2)

        ax.add_patch(rect)

    else:

        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                              fill=False, edgecolor='green', linewidth=2)

        ax.add_patch(rect)



ax.set_axis_off()

plt.tight_layout()
print("Data Available , Dimensions", em_image_vol.shape,'->',em_thresh_vol.shape)

em_label_vol, em_label_vol_num = label(em_thresh_vol, return_num = True)

print("Number of separate labels:", em_label_vol_num)
shape_analysis_list_vol = regionprops(em_label_vol)

first_region_vol = shape_analysis_list_vol[0]

print('List of region properties for',len(shape_analysis_list_vol), 'regions')

print('Features Calculated:',', '.join([f for f in dir(first_region_vol) if not f.startswith('_')]))
for prop in first_region_vol:

    try:

        print(prop, first_region_vol[prop])

    except:

        print(prop, 'is missing in 3D')
first_region_vol.filled_image.ravel().shape
from mpl_toolkits.mplot3d import Axes3D

fig =  plt.figure()

ax = fig.add_subplot(111, projection = '3d')

surf = ax.plot_surface(first_region_vol.filled_image)

plt.show()

first_region_vol.filled_image[0].shape

#x,y,z = first_region_vol.filled_image.nonzero()

#xx,yy,zz = first_region_vol.coords.nonzero()

first_region_vol.coords.shape
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

C = first_region_vol.coords

ax.plot_trisurf(C[:,0], C[:, 1], C[:,2], color='b')

plt.show()
coordsFull = np.reshape(em_image_vol, (-1, 3))

reg1 = coordsFull[first_region_vol.coords]
import pandas as pd

smooth_size = 5 # pixels

min_radius = 4

max_radius = 50
properties = []

columns = ('x', 'y', 'z', 'I', 'w')

indices = []

for z, frame in enumerate(em_label_vol):

    f_prop = regionprops(frame.astype(np.int),

                         intensity_image=em_image_vol[z])

    for d in f_prop:

        radius = (d.area / np.pi)**0.5

        if (min_radius  < radius < max_radius):

            properties.append([d.weighted_centroid[0],

                               d.weighted_centroid[1],

                               z, d.mean_intensity * d.area,

                               radius])

            indices.append(d.label)

if not len(indices):

    all_props = pd.DataFrame([], index=[])

indices = pd.Index(indices, name='label')

properties = pd.DataFrame(properties, index=indices, columns=columns)

properties['I'] /= properties['I'].max()
properties.head()