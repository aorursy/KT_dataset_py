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
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle
def detect_circles(in_img):
    edges = canny(in_img, sigma=2)
    hough_radii = np.arange(10, 90, 2)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=300)

    img1 = np.zeros(in_img.shape)
    img1 = color.gray2rgb(img1)
    for center_y, center_x, radius, (r, g, b, _) in zip(cy, cx, radii, 
                                          plt.cm.nipy_spectral(np.linspace(0,1, len(radii))) # color map
                                         ):
        circy, circx = circle(center_y, center_x, radius)
        img1[circy, circx] = (r*255, g*255, b*255)
    return img1
fig, m_axs = plt.subplots(4, 3, figsize = (20, 10))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for i, (c_path, c_ax) in enumerate(zip(scan_paths, m_axs.flatten())):
    c_img = imread(c_path)
    c_img = c_img[c_img.shape[0]//2]
    rs_img = np.clip(255*(c_img/c_img.max()), 0, 255).astype(np.uint8)
    seg_img = detect_circles(rs_img)
    stack_img = np.concatenate([plt.cm.bone(c_img/c_img.max())[:, :, :3], seg_img], 1)
    c_ax.imshow(stack_img, cmap = 'gray')
    c_ax.set_title('{:02d} - {}'.format(i, os.path.basename(c_path)))
