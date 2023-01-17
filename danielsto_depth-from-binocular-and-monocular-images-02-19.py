import os

import torch

# print(os.listdir('../input/'))

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np
# Original images

img5=mpimg.imread('../input/000005_10.png')

img11=mpimg.imread('../input/000011_10.png')



# Monodepth results

mono5=mpimg.imread('../input/000005_10_disp.png')

mono11=mpimg.imread('../input/000011_10_disp.png')



# Psmnet results

psnet5=mpimg.imread('../input/disp_kaggle.png')

psnet11=mpimg.imread('../input/disp_w.png')
from PIL import Image

img = Image.open('../input/disp_w.png').convert('P', palette='ADAPTIVE')

plt.imshow(img)
# Displaying monodepth images

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))

ax1.imshow(img5)

ax2.imshow(img5)

ax3.imshow(mono5)

ax4.imshow(psnet5)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))

ax1.imshow(img11)

ax2.imshow(img11)

ax3.imshow(mono11)

ax4.imshow(psnet11)