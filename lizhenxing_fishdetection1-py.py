import os 

from scipy import ndimage

from subprocess import check_output



import cv2

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline
img_rows, img_cols= 350, 425

im_array = cv2.imread('../input/train/LAG/img_00091.jpg',0)

template = np.zeros([ img_rows, img_cols], dtype='uint8') # initialisation of the template

template[:, :] = im_array[100:450,525:950] # I try multiple times to find the correct rectangle. 

#template /= 255.

plt.subplots(figsize=(10, 7))

plt.subplot(121),plt.imshow(template, cmap='gray') 

plt.subplot(122), plt.imshow(im_array, cmap='gray')