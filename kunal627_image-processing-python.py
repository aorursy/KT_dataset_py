# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import scipy.misc as mi 
from scipy.misc.pilutil import Image
import imageio
import matplotlib.pyplot as plt


test_image_path = '../input/test-image/download.jpg'
img_saltpepper = '../input/saltpepper/salt_pepper.png'
# read the test image
im1 = mi.imread(test_image_path)
#im1.shape
#im2 = Image.open(test_image_path)
im3 = imageio.imread(test_image_path)
#im3.shape
#mi.imshow(im1) 
# imshow throws runtime error use matplotlib instead
plt.imshow(np.uint8(im1))
import scipy.misc
from scipy.misc.pilutil import Image

# convert image to grayscale
im = Image.open(test_image_path).convert('L')

# convert to numpy ndarray
im = scipy.misc.fromimage(im)
# perform some processing and convert back to image
c = scipy.misc.toimage(im)
c
import numpy as np
import scipy.ndimage
from scipy.misc.pilutil import Image
# convert image to grayscale
a = Image.open(test_image_path).convert('L')
# 5 by 5 mean filter
k = np.ones((5,5)) / 25
# perform convolution
b = scipy.ndimage.filters.convolve(a,k)
# convert the ndarray to image
b = scipy.misc.toimage(b)
b
# mean filter removes noise and brighten the image but blurs the edges and reduces the spatial resolution
# example of median filter
import scipy.misc
import scipy.ndimage
from scipy.misc.pilutil import Image
a = Image.open(img_saltpepper).convert('L')
b = scipy.ndimage.filters.median_filter(a, size=5, footprint=None, output=None,
                                       mode='reflect', cval=0.0, origin=0)
b = scipy.misc.toimage(b)
b
# used to remove salt and pepper noise
# example of max filter
import scipy.misc
import scipy.ndimage
from scipy.misc.pilutil import Image
a = Image.open(test_image_path).convert('L')
b = scipy.ndimage.filters.maximum_filter(a, size=5, footprint=None,
                                        output=None, mode='reflect',
                                        cval=0.0, origin=0)
b = scipy.misc.toimage(b)
b
# filter enhances the bright pixels in the image
# min filter example
import scipy.misc
import scipy.ndimage
from scipy.misc.pilutil import Image
a = Image.open(test_image_path).convert('L')
b = scipy.ndimage.filters.minimum_filter(a, size=5,
        footprint=None,output=None,mode='reflect',cval=0.0,
        origin=0)
b = scipy.misc.toimage(b)
b
# minimum filter enhances the dark points in the image
import scipy.misc
from skimage import filters
from scipy.misc.pilutil import Image
a = Image.open(test_image_path).convert('L')
b = filters.sobel(a)
b = scipy.misc.toimage(b)
b
import scipy.misc
from skimage import filters
from scipy.misc.pilutil import Image
a = Image.open(test_image_path).convert('L')
b = filters.prewitt(a)
b = scipy.misc.toimage(b)
b
import scipy.misc
from skimage import filters
from scipy.misc.pilutil import Image
import matplotlib.pyplot as plt

a = Image.open(test_image_path).convert('L')
# detect horizontal edges 
b = filters.sobel_h(a)
sob_h = scipy.misc.toimage(b)
b = filters.sobel_v(a)
sob_v = scipy.misc.toimage(b)
b = filters.prewitt_v(a)
prew_v = scipy.misc.toimage(b)
b = filters.prewitt_h(a)
prew_h = scipy.misc.toimage(b)

fig, ax = plt.subplots(nrows=2, ncols=2,sharex=True)
ax[0,0].imshow(sob_h)
ax[0,0].set_title('horizontal edges Sobel')
ax[0,1].imshow(sob_v)
ax[0,1].set_title('vertical edges Sobel')
ax[1,0].imshow(prew_h)
ax[1,0].set_title('horizontal edges Prewitt')
ax[1,1].imshow(prew_v)
ax[1,1].set_title('vertical edges Prewitt')
ax[1,0].set_title('horizontal edges Prewitt')
#plt.xticks([])
#plt.yticks([])

plt.show
#canny filter
#from skimage.feature import canny
import scipy.misc
from skimage import filters
from scipy.misc.pilutil import Image
import skimage.feature as can
import matplotlib.pyplot as plt

a = Image.open(test_image_path).convert('L')
# convert to nd array
a = scipy.misc.fromimage(a)
b = can.canny(a, sigma=1)
plt.imshow(b, cmap='gray')
plt.xticks([])
plt.yticks([])

# second derivative filters
import scipy.misc
import scipy.ndimage
from scipy.misc.pilutil import Image

a = Image.open(test_image_path).convert('L')
b = scipy.ndimage.filters.laplace(a,mode='reflect')
b = scipy.misc.toimage(b)
b
# there is too much noise in the output as the noise introduced by 
#first derivative got magnified by second derivative
import scipy.misc
import scipy.ndimage
from scipy.misc.pilutil import Image
a = Image.open(test_image_path).convert('L')
b = scipy.ndimage.filters.gaussian_laplace(a, 1, mode='reflect')
b = scipy.misc.toimage(b)
b
# LoG suffers from sphageti effect. Oversegmentation of edges
# Image Enhancement
# inversion
import math
import scipy.misc
import numpy as np
from scipy.misc.pilutil import Image

im = Image.open(test_image_path).convert('L')
# convert to np array
im1 = scipy.misc.fromimage(im)
# inversion 
im2 = 255 - im1
im3 = scipy.misc.toimage(im2)
im3

# Gamma correction

import math, numpy
import scipy.misc
from scipy.misc.pilutil import Image

a = Image.open(test_image_path).convert('L')
# convert to nd array
b = scipy.misc.fromimage(a)
gamma = 2
b1 = b.astype(float)

# normalize the grayscale image array
b3 = numpy.max(b1)
b2 = b1/b3

# gamma correction
b3 = numpy.log(b2) * gamma
c = numpy.exp(b3) * 255

c1 = c.astype(int)
d = scipy.misc.toimage(c1)

d

# log transformation
import scipy.misc
import numpy, math
from scipy.misc.pilutil import Image
a = Image.open(test_image_path).convert('L')
b = scipy.misc.fromimage(a)
b1 = b.astype(float)
b2 = numpy.max(b1)
# log transformation
c = (255.0 * numpy.log(1 + b1)) / numpy.log(1+b2)
c1 = c.astype(int)
d = scipy.misc.toimage(c1)
d
# histogram equalization
import numpy as np
import scipy.misc, math
from scipy.misc.pilutil import Image
img = Image.open(test_image_path).convert('L')
img1 = scipy.misc.fromimage(img)
fl = img1.flatten()
hist, bins = np.histogram(img1, 256, [0,255])
cdf = hist.cumsum()
# mask the places where cdf - 0
cdf_m = np.ma.masked_equal(cdf,0)
# histogram equalization
num_cdf_m = (cdf_m - cdf_m.min())
den_cdf_m = (cdf_m.max() - cdf_m.min())
cdf_m = num_cdf_m/den_cdf_m
# the masked places in cdf_m are now 0
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
im2 = cdf[fl]
im3 = np.reshape(im2, img1.shape)
im4 = scipy.misc.toimage(im3)
im4
#contrast streatching
# similar to histogram equalization except that pixel intensities are related using the pixel values 
# instead of probabilities and cdf
import math, numpy
import scipy.misc
from scipy.misc.pilutil import Image
im = Image.open(test_image_path).convert('L')
im1 = scipy.misc.fromimage(im)
b = im1.max()
a = im2.min()
c = im1.astype(float)
im2 = 255 * (c - a) /(b-a)
im3 = scipy.misc.toimage(im2)
im3
import math, numpy
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image
a = Image.open(img_saltpepper).convert('L')
b = numpy.asarray(a)
# perform fft
c = abs(fftim.fft2(b))
# shift the fourier frequency image
d = fftim.fftshift(c)
d.astype('float').tofile('fft.raw')
