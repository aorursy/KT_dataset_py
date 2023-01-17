%matplotlib inline
import numpy as np
from skimage.io import imread
from glob import glob
import os
import matplotlib.pyplot as plt
base_dir = os.path.join('..', 'input')
scan_paths = sorted(glob(os.path.join(base_dir, '*.tif'))) # assume scans are ordered by time?
print(len(scan_paths), 'scans found')
fig, m_axs = plt.subplots(4, 3, figsize = (20, 12))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for i, (c_path, c_ax) in enumerate(zip(scan_paths, m_axs.flatten())):
    c_img = imread(c_path)
    c_ax.imshow(c_img[c_img.shape[0]//2], cmap = 'gray')
    c_ax.set_title('{:02d} - {}'.format(i, os.path.basename(c_path)))
fig, m_axs = plt.subplots(4, 3, figsize = (20, 12))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for i, (c_path, c_ax) in enumerate(zip(scan_paths, m_axs.flatten())):
    c_img = imread(c_path)
    c_ax.imshow(np.sum(c_img,0), cmap = 'gray')
    c_ax.set_title('{:02d} - {}'.format(i, os.path.basename(c_path)))
fig, m_axs = plt.subplots(4, 3, figsize = (20, 10))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for i, (c_path, c_ax) in enumerate(zip(scan_paths, m_axs.flatten())):
    c_img = imread(c_path)
    c_ax.imshow(c_img[:, c_img.shape[1]//2, :], cmap = 'gray')
    c_ax.set_title('{:02d} - {}'.format(i, os.path.basename(c_path)))
fig, m_axs = plt.subplots(4, 3, figsize = (20, 10))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for i, (c_path, c_ax) in enumerate(zip(scan_paths, m_axs.flatten())):
    c_img = imread(c_path)
    c_ax.imshow(c_img[:, :, c_img.shape[2]//2], cmap = 'gray')
    c_ax.set_title('{:02d} - {}'.format(i, os.path.basename(c_path)))
from skimage.filters import threshold_isodata
from skimage.morphology import label
from skimage.segmentation import clear_border, mark_boundaries
def seg_foam(in_img):
    seg_img = c_img<threshold_isodata(c_img) # bubles are negative phase
    # pad top and bottom slices before clearing edges
    pad_seg = np.pad(seg_img, [(1,1), (0,0), (0,0)], mode = 'constant', constant_values = 1)
    return clear_border(pad_seg)[1:-1] # remove padding and border
fig, m_axs = plt.subplots(4, 3, figsize = (20, 10))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for i, (c_path, c_ax) in enumerate(zip(scan_paths, m_axs.flatten())):
    c_img = imread(c_path)
    seg_img = seg_foam(c_img)
    c_ax.imshow(np.sum(seg_img, 2), cmap = 'gray')
    c_ax.set_title('{:02d} - {}'.format(i, os.path.basename(c_path)))
