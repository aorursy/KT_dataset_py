%matplotlib inline 

# just once to setup the plots in the notebook (% is for 'magic' commands in jupyter)



import numpy as np # linear algebra

from skimage.io import imread # for reading images

import matplotlib.pyplot as plt # for showing figures
bone_image = imread('../input/bone.tif')

print('Loading bone image shape: {}'.format(bone_image.shape))
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 4))

ax1.imshow(bone_image, cmap = 'bone')

_ = ax2.hist(bone_image.ravel(), 20)
silly_thresh_value = 55

thresh_image = bone_image > silly_thresh_value



fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20, 10))

ax1.imshow(bone_image, cmap = 'bone')

ax1.set_title('Original Image')

ax2.imshow(thresh_image, cmap = 'jet')

ax2.set_title('Thresheld Image')
# import the needed morphological operations

from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing

## your code
# import a few filters

from skimage.filters import gaussian, median

## your code
threshold_list = [10, 20, 200]



fig, m_ax = plt.subplots(2, len(threshold_list), figsize = (15, 6))

for c_thresh, (c_ax1, c_ax2) in zip(threshold_list, m_ax.T):

    bone_thresh = bone_image > c_thresh

    # your code here

    c_ax1.imshow(bone_thresh, cmap = 'jet')

    c_ax1.set_title('Bone @ {}, Image'.format(c_thresh))

    c_ax1.axis('off')

    

    # do cells

    cell_thresh = bone_image < c_thresh

    # your code here

    c_ax2.imshow(cell_thresh, cmap = 'jet')

    c_ax2.set_title('Cell @ {}, Image'.format(c_thresh))

    c_ax2.axis('off')

    