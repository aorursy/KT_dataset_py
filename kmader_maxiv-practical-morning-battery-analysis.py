!ls ../input -R 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
from glob import glob
all_files = sorted(glob('../input/nmc_90wt_0bar/NMC_90wt_0bar/grayscale/*.tif'))
print(len(all_files), all_files[0])
from skimage.io import imread
from skimage import img_as_float
first_image = imread(all_files[0])
first_image_float = img_as_float(first_image)
print(first_image_float.shape)
import matplotlib.pyplot as plt
plt.imshow(first_image_float, cmap = 'gray')
fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
img_plot_view = ax1.imshow(first_image_float[300:700, 300:700], cmap = 'gray')
plt.colorbar(img_plot_view)
print(np.eye(3))
print(np.eye(3).ravel())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.hist(first_image_float[300:700, 300:700].ravel(), 50)
img_plot_view = ax2.imshow(first_image_float[300:700, 300:700], cmap = 'gray')
plt.colorbar(img_plot_view)
middle_image = imread(all_files[110])
middle_image_float = img_as_float(middle_image)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.hist(middle_image_float[300:700, 300:700].ravel(), 50)
img_plot_view = ax2.imshow(middle_image_float[300:700, 300:700], cmap = 'gray')
plt.colorbar(img_plot_view)
from skimage.measure import regionprops
from skimage.morphology import label
def try_threshold(thresh_val):
    roi_img = middle_image_float[300:700, 300:700]
    seg_img = roi_img>thresh_val
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (21, 7))
    ax1.hist(roi_img.ravel(), 50)
    ax1.axvline(thresh_val, color = 'red')
    ax1.set_title('Intensity Distribution')
    ax2.imshow(seg_img)
    ax2.set_title('Threshold Image')
    ax3.hist([c_reg.major_axis_length for c_reg in regionprops(label(seg_img))])
    ax3.set_title('Object Diameters')
    img_plot_view = ax4.imshow(roi_img, cmap = 'gray')
    ax4.set_title('ROI Image')
    plt.colorbar(img_plot_view)
    fig.savefig('thresh_image.pdf')
    return seg_img
try_threshold(0.5);
from skimage.filters import try_all_threshold
try_all_threshold(middle_image_float[300:700, 300:700])
import skimage
skimage.filters.thresholding.threshold_isodata(middle_image_float[300:700, 300:700])
from skimage.morphology import label
seg_img = try_threshold(0.5)
lab_img = label(seg_img)
print('Objects Found', lab_img.max()+1)
plt.imshow(lab_img, cmap = 'jet')
test_thresh_vals = np.linspace(0.3, 0.6, 10)
obj_count = []
for c_thresh_val in test_thresh_vals:
    seg_img = try_threshold(c_thresh_val)
    lab_img = label(seg_img)
    print('Objects Found', lab_img.max()+1)
    obj_count += [lab_img.max()+1]
plt.plot(test_thresh_vals, obj_count, 'rs-')
from skimage.segmentation import mark_boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7))
ax1.imshow(lab_img, cmap = 'jet')
ax2.imshow(mark_boundaries(middle_image_float[300:700, 300:700], label_img=lab_img))
plt.imshow(mark_boundaries(np.zeros_like(lab_img), label_img=lab_img))
from scipy.ndimage import distance_transform_edt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7))
ax1.imshow(seg_img, cmap = 'jet')
dist_plot = ax2.imshow(distance_transform_edt(seg_img), cmap = 'jet', vmin = 0, vmax = 10)
plt.colorbar(dist_plot)
from skimage.measure import regionprops
for c_reg in regionprops(lab_img):
    print(c_reg.major_axis_length)
