import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage import io, filters, color, data
import glob
files = glob.glob("../input/*/*/*/*")
plt.figure(figsize=(20,50))

for i in range(10):
    im = io.imread(files[i])
    img = color.rgb2gray(im)

    plt.subplot(10,2,i*2+1); plt.imshow(im)
    plt.subplot(10,2,i*2+2); plt.imshow(img, cmap="gray")
plt.figure(figsize=(20,30))
img_sobel = filters.sobel(img)
plt.imshow(1-img_sobel, cmap="gray")
im = data.astronaut()
plt.imshow(im)
im_hsv = color.convert_colorspace(im, 'RGB', 'HSV')
im_rgbcie = color.convert_colorspace(im, 'RGB', 'RGB CIE')
im_xyz = color.convert_colorspace(im, 'RGB', 'XYZ')
im_yuv = color.convert_colorspace(im, 'RGB', 'YUV')
im_yiq = color.convert_colorspace(im, 'RGB', 'YIQ')
plt.imshow(im_hsv)
plt.imshow(im_rgbcie)
plt.imshow(im_xyz)
plt.imshow(im_4)
plt.imshow(im_5)
