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
import matplotlib.pyplot as plt
plt.imshow(first_image_float, cmap = plt.cm.ocean, vmin = 0, vmax = 0.5)
fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
img_plot_view = ax1.imshow(first_image_float[700:1000, 300:700], cmap = plt.cm.Purples)
plt.colorbar(img_plot_view)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.hist(first_image_float[300:700, 300:700].ravel(), 50)
img_plot_view = ax2.imshow(first_image_float[300:700, 300:700], cmap = 'gray')
plt.colorbar(img_plot_view)
middle_image = imread(all_files[110])
middle_image_float = img_as_float(middle_image)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.hist(middle_image_float[700:1000, 300:700].ravel(), 50)
img_plot_view = ax2.imshow(middle_image_float[700:1000, 300:700], cmap = 'gray')
plt.colorbar(img_plot_view)
roi_image = middle_image_float[700:1000, 300:700]
plt.imshow(roi_image>0.5)
import seaborn as sns
sns.heatmap(roi_image[:20:2, :20:2], annot = True)

sns.heatmap(roi_image[:20:2, :20:2]>0.6, annot = True)
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
try_all_threshold(roi_image, figsize = (10, 20))
import skimage
skimage.filters.thresholding.threshold_yen(middle_image_float)
from skimage.morphology import label
seg_img = try_threshold(0.55)
lab_img = label(seg_img)
print('Objects Found', lab_img.max()+1)
from skimage.segmentation import mark_boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7))
ax1.imshow(lab_img, cmap = 'jet')
ax2.imshow(mark_boundaries(middle_image_float[300:700, 300:700], label_img=lab_img))
from skimage.measure import regionprops
all_radii = []
for c_reg in regionprops(lab_img):
    all_radii += [c_reg.major_axis_length*0.37]
plt.hist(all_radii, 20)
np.mean(all_radii)
big_particles = []
nd_isin = lambda x, ids: np.isin(x.ravel(), ids).reshape(x.shape)
for c_reg in regionprops(lab_img):
    if c_reg.major_axis_length*0.37>20:
        big_particles += [c_reg.label]
plt.imshow(nd_isin(lab_img, big_particles))
# comparing to second sample at 2000bar
hp_files = sorted(glob('../input/nmc_90wt_2000bar/NMC_90wt_2000bar/grayscale/*.tif'))
print(len(hp_files), hp_files[0])
hp_slice = imread(hp_files[110])
hp_slice_float = img_as_float(hp_slice)
hp_slice_float
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.hist(hp_slice_float[700:1000, 300:700].ravel(), 50)
img_plot_view = ax2.imshow(hp_slice_float[700:1000, 300:700], cmap = 'gray')
plt.colorbar(img_plot_view)
seg_hp_img = hp_slice_float[700:1000, 300:700]>0.55
lab_hp_img = label(seg_hp_img)
print('Objects Found', lab_hp_img.max()+1)
from skimage.measure import regionprops
all_hp_radii = []
for c_reg in regionprops(lab_hp_img):
    all_hp_radii += [c_reg.major_axis_length*0.37]
plt.hist(all_hp_radii, 20)
fig, ax1 = plt.subplots(1,1, figsize = (6, 6))
ax1.hist(all_radii, np.linspace(0, 30, 20), label = '0 bar')
ax1.hist(all_hp_radii, np.linspace(0, 30, 20), label = '2000 bar', alpha = 0.5)
ax1.legend()
from scipy.stats import ttest_ind
ttest_ind(all_radii, all_hp_radii)
