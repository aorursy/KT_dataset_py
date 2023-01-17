!ls ../input/train-tif-v2/ | head
import os



import numpy as np

from spectral import *

from skimage import io



import matplotlib.pyplot as plt

%matplotlib inline
BASEPATH = os.path.abspath('../input/train-tif-v2/')
path = os.path.join(BASEPATH, 'train_1.tif')

img = io.imread(path)

bgr = img[:,:,:3]

rgb = bgr_image[:, :, [2,1,0]]

plt.imshow(rgb)
# note: the error below is an issue with spectral

# running w/python 3.

# See https://github.com/spectralpython/spectral/issues/56

# Normally it shows stats about the image. BUT at least

# the image renders more how you'd expect :)

imshow(rgb)