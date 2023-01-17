# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input/train/c0"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters,feature
from skimage.color import rgb2gray
from skimage import io,transform
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
%matplotlib inline

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

def as_gray(image_filter, image, *args, **kwargs):
    gray_image = rgb2gray(image)
    return image_filter(gray_image, *args, **kwargs)

@adapt_rgb(as_gray)
def sobel_gray(image):
    return filters.sobel(image)


im = io.imread('../input/train/c0/img_100026.jpg')
plt.imshow(im)
img = transform.resize(im,(120,160))
rgb_sobel = rescale_intensity(1 - sobel_each(img))
hsv_sobel = rescale_intensity(1 - sobel_hsv(img))
grey_sobel = rescale_intensity(1 - sobel_gray(img))
rgb_hsv_sobel = rescale_intensity(1 - sobel_hsv(rgb_sobel))
rgb_gray_sobel = rescale_intensity(1 - sobel_gray(rgb_sobel))
hsv_gray_sobel = rescale_intensity(1 - sobel_gray(hsv_sobel))

fig = plt.figure(1,figsize=(10,15))
plt.subplot(131)
plt.imshow(rgb_sobel)



plt.subplot(133)
plt.imshow(feature.canny(grey_sobel))
hsv_sobel = rescale_intensity(1 - sobel_hsv(grey_sobel))
plt.subplot(132)
plt.imshow(hsv_sobel)
plt.imshow(rgb2gray(im))