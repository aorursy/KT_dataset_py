from skimage import io

from skimage import color

import matplotlib.pyplot as plt

from skimage import data

from pylab import *

%matplotlib inline
image = io.imread('../input/dog-image/puppy.jpg')
image_hsv = color.rgb2hsv(image)

image_rgb = color.hsv2rgb(image_hsv)
figure(0)

plt.figure(figsize = (10, 8))

plt.title("HSV image")

io.imshow(image_hsv)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)
image_xyz = color.rgb2xyz(image)

image_rgb = color.xyz2rgb(image_xyz)


figure(0)

plt.figure(figsize = (10, 8))

plt.title("XYZ image")

io.imshow(image_xyz)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)
image_lab = color.rgb2lab(image)

image_rgb = color.lab2rgb(image_xyz)
figure(0)

plt.figure(figsize = (10, 8))

plt.title("LAB image")

io.imshow(image_lab)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)
image_yuv = color.rgb2yuv(image)

image_rgb = color.yuv2rgb(image_yuv)
figure(0)

plt.figure(figsize = (10, 8))

plt.title("YUV image")

io.imshow(image_yuv)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)
image_yiq = color.rgb2yiq(image)

image_rgb = color.yiq2rgb(image_yiq)
figure(0)

plt.figure(figsize = (10, 8))

plt.title("YIQ image")

io.imshow(image_yiq)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)
image_gray = color.rgb2gray(image)

image_rgb = color.gray2rgb(image_gray)
figure(0)

plt.figure(figsize = (10, 8))

plt.title("GRAY image")

io.imshow(image_gray)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)
image_hed = color.rgb2hed(image)

image_rgb = color.hed2rgb(image_hed)
figure(0)

plt.figure(figsize = (10, 8))

plt.title("HED image")

io.imshow(image_hed)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)
image_rgbcie = color.rgb2rgbcie(image)

image_rgb = color.rgbcie2rgb(image_rgbcie)
figure(0)

plt.figure(figsize = (10, 8))

plt.title("RGBIC image")

io.imshow(image_rgbcie)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)
image_ycbcr = color.rgb2ycbcr(image)

image_rgb = color.ycbcr2rgb(image_ycbcr)
figure(0)

plt.figure(figsize = (10, 8))

plt.title("YCBCR image")

io.imshow(image_ycbcr)

figure(1)

plt.figure(figsize = (10, 8))

plt.title("RGB image")

io.imshow(image_rgb)