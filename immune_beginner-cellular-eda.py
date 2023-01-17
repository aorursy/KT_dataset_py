# 🎨 Justin Faler 

# 📆 8/30/2019



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from skimage.io import imread, imshow, imsave

from skimage.filters import prewitt_h,prewitt_v

from skimage.color import rgb2hsv

import scipy.misc

import scipy.ndimage

import sklearn.metrics

from sklearn.cluster import KMeans

from skimage import measure

import imageio

import cv2

import skimage.io

import os
df = pd.read_csv('../input/recursion-cellular-image-classification/pixel_stats.csv')

df2 = pd.read_csv("../input/recursion-cellular-image-classification/test_controls.csv")

df3 = pd.read_csv('../input/recursion-cellular-image-classification/train_controls.csv')

train = pd.read_csv('../input/recursion-cellular-image-classification/train.csv')

test = pd.read_csv('../input/recursion-cellular-image-classification/test.csv')
print(train.head())
print(df.head())
print(df2.head())
print(df3.head())
print(test.head())
test_img_f = '../input/recursion-cellular-image-classification/test/HUVEC-21/Plate3/H05_s2_w3.png'

im = skimage.io.imread(test_img_f)

im_g = skimage.io.imread(test_img_f, as_gray=True)



#skimage.io.imshow(im)

im.dtype
plt.figure(figsize=(15, 15))

image = imread('../input/recursion-cellular-image-classification/test/HUVEC-21/Plate3/H05_s2_w3.png', as_gray=True)

imshow(image)

plt.ylabel('Height {}'.format(image.shape[0]))

plt.xlabel('Width {}'.format(image.shape[1]))
# Photo negative

plt.figure(figsize=(20, 15))

negative = 255 - image # neg = (L-1) - img

plt.ylabel('Height {}'.format(image.shape[0]))

plt.xlabel('Width {}'.format(image.shape[1]))

plt.imshow(negative);
plt.figure(figsize=(15, 15))

pic = imageio.imread('../input/recursion-cellular-image-classification/test/HUVEC-21/Plate3/H05_s2_w3.png')



h,w = pic.shape[:2]



im_small_long = pic.reshape((h * w, 1))

im_small_wide = im_small_long.reshape((h,w,1))



km = KMeans(n_clusters=2)

km.fit(im_small_long)



seg = np.asarray([(1 if i == 1 else 0)

                  for i in km.labels_]).reshape((h,w))



contours = measure.find_contours(seg, 0.5, fully_connected="high")

simplified_contours = [measure.approximate_polygon(c, tolerance=5) 

                       for c in contours]



plt.figure(figsize=(20,15))

for n, contour in enumerate(simplified_contours):

    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

    

    

plt.ylim(h,0)

# Matrix 🔴🆚🔵💊

image = imread('../input/recursion-cellular-image-classification/test/HUVEC-21/Plate3/H05_s2_w3.png')

image.shape, image
df.values
df2 = pd.DataFrame(np.random.rand(10, 11), columns=['id_code', 'experiment', 'mean', 'std', 'plate','well', 'site', 'channel', 'median', 'min', 'max'])

df2.plot.bar(figsize=(20,15));
corr = df2.corr()

fig = plt.figure(1, figsize=(20,15))

plt.imshow(corr,cmap='winter')

labels = np.arange(len(df2.columns))

plt.xticks(labels,df2.columns,rotation=90)

plt.yticks(labels,df2.columns)

plt.title('Correlation Matrix of Global Variables')

cbar = plt.colorbar(shrink=0.85,pad=0.02)

plt.show()
df.hist(bins=50, figsize=(20,15))

plt.show()
plt.figure(); df.plot(figsize=(15,10))
plt.figure(); df2.plot(figsize=(15,10))
sns.pairplot(df2,hue='channel',height=2.6)
df.loc[0]
image = imread('../input/recursion-cellular-image-classification/test/HUVEC-21/Plate3/H05_s2_w3.png')

print('Type of the image : ' , type(image))



print('Shape of the image : {}'.format(image.shape))



print('Image Hight {}'.format(image.shape[0]))



print('Image Width {}'.format(image.shape[1]))



print('Dimension of Image {}'.format(image.ndim))
image = imread('../input/recursion-cellular-image-classification/test/HUVEC-21/Plate3/H05_s2_w3.png')

print('Image size {}'.format(image.size))



print('Maximum RGB value in this image {}'.format(image.max()))



print('Minimum RGB value in this image {}'.format(image.min()))
df.shape
plt.figure(figsize=(15, 15))

grayscale = imread('../input/recursion-cellular-image-classification/test/HUVEC-21/Plate3/H05_s2_w3.png')

counts, vals = np.histogram(grayscale, bins=range(2 ** 8))

plt.plot(range(0, (2 ** 8) - 1), counts)

plt.title('Grayscale image histogram')

plt.xlabel('Pixel intensity')

plt.ylabel('Count')
df.size
df.isnull().sum()
# look at the last 5 rows

df.tail()
df.describe()

print("*"*50)

df.info()

print("*"*50)
df2