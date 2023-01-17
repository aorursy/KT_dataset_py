# 🎨 Justin Faler 

# 📆 8/30/2019

# 🦅 Mt. San Jacinto College



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from skimage.io import imread, imshow, imsave

import cv2 # opencv version 3.4.2

from skimage.filters import prewitt_h,prewitt_v

from skimage.color import rgb2hsv

import scipy.misc

import scipy.ndimage

import sklearn.metrics

from sklearn.cluster import KMeans

import matplotlib as mpl

from skimage import measure

import imageio

import os
df = pd.read_csv('../input/open-images-2019-object-detection/sample_submission.csv')
print('# File sizes')

for f in os.listdir('../input'):

    if not os.path.isdir('../input/' + f):

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')

    else:

        sizes = [os.path.getsize('../input/'+f+'/'+x)/1000000 for x in os.listdir('../input/' + f)]

        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))
df.values
import skimage

import skimage.io



test_img_f = '../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg'

im = skimage.io.imread(test_img_f)

im_g = skimage.io.imread(test_img_f, as_gray=True)



#skimage.io.imshow(im)

im.dtype
image = imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg', as_gray=True)

imshow(image)

plt.ylabel('Height {}'.format(image.shape[0]))

plt.xlabel('Width {}'.format(image.shape[1]))
image = imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg')

print('Type of the image : ' , type(image))



print('Shape of the image : {}'.format(image.shape))



print('Image Hight {}'.format(image.shape[0]))



print('Image Width {}'.format(image.shape[1]))



print('Dimension of Image {}'.format(image.ndim))
image = imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg')

print('Image size {}'.format(image.size))



print('Maximum RGB value in this image {}'.format(image.max()))



print('Minimum RGB value in this image {}'.format(image.min()))
# RGB to HSV(Hue, Saturation, Value)

inp_image = imread("../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg")

hsv_img = rgb2hsv(inp_image)

plt.ylabel('Height {}'.format(image.shape[0]))

plt.xlabel('Width {}'.format(image.shape[1]))

imshow(hsv_img)
grayscale = imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg')

counts, vals = np.histogram(grayscale, bins=range(2 ** 8))

plt.plot(range(0, (2 ** 8) - 1), counts)

plt.title('Grayscale image histogram')

plt.xlabel('Pixel intensity')

plt.ylabel('Count')
pic = imageio.imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg')



h,w = pic.shape[:2]



im_small_long = pic.reshape((h * w, 3))

im_small_wide = im_small_long.reshape((h,w,3))



km = KMeans(n_clusters=2)

km.fit(im_small_long)



seg = np.asarray([(1 if i == 1 else 0)

                  for i in km.labels_]).reshape((h,w))



contours = measure.find_contours(seg, 0.5, fully_connected="high")

simplified_contours = [measure.approximate_polygon(c, tolerance=5) 

                       for c in contours]



plt.figure(figsize=(5,10))

for n, contour in enumerate(simplified_contours):

    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

    

    

plt.ylim(h,0)

plt.axes().set_aspect('equal')

image = imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg')



'''

Let's pick a specific pixel located at 100 th Rows and 50 th Column. 

And view the RGB value gradually. 

'''



image[ 100, 50 ]
image = imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg')

# A specific pixel located at Row : 100 ; Column : 50 

# Each channel's value of it, gradually R , G , B



print('Value of only R channel {}'.format(image[ 100, 50, 0]))



print('Value of only G channel {}'.format(image[ 100, 50, 1]))



print('Value of only B channel {}'.format(image[ 100, 50, 2]))
plt.title('R channel')



plt.ylabel('Height {}'.format(image.shape[0]))



plt.xlabel('Width {}'.format(image.shape[1]))



plt.imshow(image[ : , : , 0])



plt.show()
plt.title('G channel')



plt.ylabel('Height {}'.format(image.shape[0]))



plt.xlabel('Width {}'.format(image.shape[1]))



plt.imshow(image[ : , : , 1])



plt.show()
plt.title('B channel')



plt.ylabel('Height {}'.format(image.shape[0]))



plt.xlabel('Width {}'.format(image.shape[1]))



plt.imshow(image[ : , : , 2])



plt.show()
# Photo negative

negative = 255 - image # neg = (L-1) - img

plt.ylabel('Height {}'.format(image.shape[0]))

plt.xlabel('Width {}'.format(image.shape[1]))

plt.imshow(negative);
# Here is that cup in a matrix 🔴🆚🔵💊

image = imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg')

image.shape, image
image = imread('../input/open-images-2019-object-detection/test/1ae704327598297d.jpg') 

image.shape
# Lets generate some edge detection

image = imread('../input/open-images-2019-object-detection/test/d0d394a4b854c49d.jpg',as_gray=True)



#calculating horizontal edges using prewitt kernel

edges_prewitt_horizontal = prewitt_h(image)

#calculating vertical edges using prewitt kernel

edges_prewitt_vertical = prewitt_v(image)



imshow(edges_prewitt_vertical, cmap='gray')
# Print the size of the sample submission

df.size
# Print the shape 

df.shape
df
df.loc[0]
df.isnull().sum()
df.head
# look at the last 5 rows

df.tail()
df.describe()

print("*"*50)

df.info()

print("*"*50)
df.dtypes