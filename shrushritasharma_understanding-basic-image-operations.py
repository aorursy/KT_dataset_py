# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing essential libraries

import cv2, glob, os

from matplotlib import pyplot as plt

from PIL import Image
# Importing essential datasets

BASE = '../input/human-faces/Humans/'
# Reading and understanding a single image

img = cv2.imread("../input/human-faces/Humans/1 (1010).jpg")

plt.figure(figsize=(10,10))

plt.imshow(img, cmap = 'twilight'), plt.axis('off'), plt.title('Good guy?',fontsize=20),plt.show()
# Loading a whole bunch of images in train and test datasets

#image_train = [cv2.imread(file) for file in glob.glob(BASE+'*.jpg')]

#image_test = [cv2.imread(file) for file in glob.glob(BASE+'*.png')]

image_01 = cv2.imread(BASE+'1 (3443).jpg')

image_02 = cv2.imread(BASE+'1 (3043).jpg')
# Visualizing the selected images

plt.figure(figsize=(20,10))

plt.subplot(121),plt.imshow(image_01, cmap = 'hsv'), plt.axis('off'), plt.title('Nice lady',fontsize=30)

plt.subplot(122),plt.imshow(image_02, cmap = 'cividis'), plt.axis('off'), plt.title('Simple girl',fontsize=30)

plt.show()
# Visualizing the image histogram for first image

counts,bins,_ = plt.hist(image_01.ravel(),density = False, alpha = 0.8, histtype = 'stepfilled', color = '#0303FF', edgecolor = '#44FF80')
# Visualizing the histogram for second image

counts,bins,_ = plt.hist(image_02.ravel(),density = True, alpha = 0.2, histtype = 'stepfilled', color = '#03DDFF', edgecolor = '#FF0000')
# Understanding multivariate normal for the first image

x, y = np.random.multivariate_normal([0,200],[[1, 0], [0, 200]],10000).T

plt.hist2d(x,y,bins=30,cmap="Blues")

cb = plt.colorbar()

cb.set_label('Counts in Bin')

plt.show()
# Understanding multivariate normal for the second image

x, y = np.random.multivariate_normal([0,200],[[1, 0], [0, 200]],10000).T

plt.hist2d(x,y,bins=30,cmap="Greens")

cb = plt.colorbar()

cb.set_label('Counts in Bin')

plt.show()
# Grayscale histogram

plt.figure(figsize=(15,8))

plt.subplot(241), plt.plot(cv2.calcHist([cv2.cvtColor(image_01, cv2.COLOR_BGR2GRAY)],[0],None,[256], [0,256]), color = 'k'), plt.title('Nice lady',fontsize=15)

plt.subplot(242), plt.plot(cv2.calcHist([image_01],[0],None,[256],[0,256]),color = 'b'), plt.xlim([0,256])

plt.subplot(243), plt.plot(cv2.calcHist([image_01],[0],None,[256],[0,256]),color = 'g'), plt.xlim([0,256])

plt.subplot(244), plt.plot(cv2.calcHist([image_01],[0],None,[256],[0,256]),color = 'r'), plt.xlim([0,256])

plt.subplot(245), plt.plot(cv2.calcHist([cv2.cvtColor(image_02, cv2.COLOR_BGR2GRAY)],[0],None,[256], [0,256]), color = 'k'), plt.title('Cute Girl',fontsize=15)

plt.subplot(246), plt.plot(cv2.calcHist([image_02],[0],None,[256],[0,256]),color = 'b'), plt.xlim([0,256])

plt.subplot(247), plt.plot(cv2.calcHist([image_02],[0],None,[256],[0,256]),color = 'g'), plt.xlim([0,256])

plt.subplot(248), plt.plot(cv2.calcHist([image_02],[0],None,[256],[0,256]),color = 'r'), plt.xlim([0,256])

plt.show()
# Grayscale Histogram Equalization

plt.figure(figsize=(20,10))

plt.subplot(121),plt.imshow(cv2.cvtColor(image_01, cv2.COLOR_BGR2GRAY), cmap = 'gray'), plt.axis('off'), plt.title('Nice lady',fontsize=20)

plt.subplot(122),plt.imshow(cv2.equalizeHist(cv2.cvtColor(image_01, cv2.COLOR_BGR2GRAY)), cmap = 'gray'), plt.axis('off'), plt.title('Equalized Histogram',fontsize=20)

plt.show()
# 3-channel Histogram Equalization

channels = cv2.split(image_01)

eq_channels = []

for ch, color in zip(channels, ['B', 'G', 'R']): 

    eq_channels.append(cv2.equalizeHist(ch))

plt.figure(figsize=(20,10))

plt.subplot(121),plt.imshow(image_01, cmap = 'gray'), plt.axis('off'), plt.title('Nice lady',fontsize=20)

plt.subplot(122),plt.imshow(cv2.cvtColor(cv2.merge(eq_channels),cv2.COLOR_BGR2RGB), cmap = 'gray'), plt.axis('off'), plt.title('Equalized Histogram',fontsize=20)

plt.show()
# Averaging the images

plt.figure(figsize=(20,10))

plt.subplot(121),plt.imshow(cv2.blur(image_01,(40,40)), cmap = 'hsv'), plt.axis('off'), plt.title('Nice lady',fontsize=30)

plt.subplot(122),plt.imshow(cv2.blur(image_02,(20,20)), cmap = 'cividis'), plt.axis('off'), plt.title('Cute girl',fontsize=30)

plt.show()
# Gaussian filtering the images

plt.figure(figsize=(20,10))

plt.subplot(121), plt.imshow(cv2.GaussianBlur(image_01,(5,5),0), cmap = 'hsv'), plt.axis('off'), plt.title('Nice lady',fontsize=30)

plt.subplot(122), plt.imshow(cv2.GaussianBlur(image_02,(5,5),0), cmap = 'hsv'), plt.axis('off'), plt.title('Cute girl',fontsize=30)

plt.show()
# Median filtering the images

plt.figure(figsize=(20,10))

plt.subplot(121),plt.imshow(cv2.medianBlur(image_01,5), cmap = 'hsv'), plt.axis('off'), plt.title('Nice lady',fontsize=30)

plt.subplot(122),plt.imshow(cv2.medianBlur(image_02,5), cmap = 'Blues'), plt.axis('off'), plt.title('Cute girl',fontsize=30)

plt.show()
# Bilateral filtering the images

plt.figure(figsize=(20,10))

plt.subplot(121),plt.imshow(cv2.bilateralFilter(image_01,9,7.5,7.5), cmap = 'hsv'), plt.axis('off'), plt.title('Nice lady',fontsize=30)

plt.subplot(122),plt.imshow(cv2.bilateralFilter(image_02,9,7.5,7.5), cmap = 'Blues'), plt.axis('off'), plt.title('Cute girl',fontsize=30)

plt.show()
# ROI selection in image the images

# image_01[300:600,170:400] where the first is from top to bottom and the second is from left to right

eye_01 = image_01[380:480,180:390]

eye_02 = image_02[420:500,190:310]

plt.figure(figsize=(20,10))

plt.subplot(121), plt.imshow(eye_01, cmap = 'hsv'), plt.axis('off'), plt.title('Nice Lady Eye',fontsize=30)

plt.subplot(122), plt.imshow(eye_02, cmap = 'Blues'), plt.axis('off'), plt.title('Cute Girl Eye',fontsize=30)

plt.show()
# Randomly getting some image information

print(image_01.shape)

print(image_01.dtype)

print(eye_01.shape)

print(eye_02.shape)
# Making borders for the images

plt.figure(figsize=(20,10))

plt.subplot(231), plt.imshow(eye_01, cmap = 'gray'), plt.axis('off'), plt.title('Grey',fontsize=25)

plt.subplot(232), plt.imshow(cv2.copyMakeBorder(eye_01,10,10,10,10,cv2.BORDER_REPLICATE), cmap = 'Blues'), plt.axis('off'), plt.title('Replicate',fontsize=25)

plt.subplot(233), plt.imshow(cv2.copyMakeBorder(eye_01,10,10,10,10,cv2.BORDER_REFLECT), cmap = 'gray'), plt.axis('off'), plt.title('Reflect',fontsize=25)

plt.subplot(234), plt.imshow(cv2.copyMakeBorder(eye_01,10,10,10,10,cv2.BORDER_REFLECT_101), cmap = 'Blues'), plt.axis('off'), plt.title('Reflect 101',fontsize=25)

plt.subplot(235), plt.imshow(cv2.copyMakeBorder(eye_01,10,10,10,10,cv2.BORDER_WRAP), cmap = 'gray'), plt.axis('off'), plt.title('Wrap',fontsize=25)

plt.subplot(236), plt.imshow(cv2.copyMakeBorder(eye_01,10,10,10,10,cv2.BORDER_CONSTANT,value=(120,80,250)), cmap = 'Blues'), plt.axis('off'), plt.title('Constant',fontsize=25)

plt.subplots_adjust(wspace=0.05, hspace=-0.3)

plt.show()
# Mask operations for the images

kernel = cv2.getGaussianKernel(15, 2.0)

kernel_2D = kernel @ kernel.transpose()

blurred_eye = cv2.filter2D(eye_01, -1, kernel_2D)

plt.imshow(blurred_eye, cmap = 'Blues'), plt.axis('off'), plt.title('Gaussian masking',fontsize=20), plt.show()
# Blending images

#dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])

#dst = src1 * alpha + src2 * beta + gamma

eye_02x = cv2.resize(eye_02,eye_01.shape[1::-1])

blended_image = cv2.addWeighted(eye_01, 0.5, eye_02x, 0.5, 0)

plt.imshow(blended_image, cmap = 'Blues'), plt.axis('off'), plt.title('Merging images using weights',fontsize=20), plt.show()
# Masking images

plt.imshow(cv2.bitwise_and(eye_01, eye_02x), cmap = 'Blues'), plt.axis('off'), plt.title('Bitwise masking',fontsize=20), plt.show()
# Masking images

plt.imshow((eye_01*0.2+eye_02x*0.8).astype(np.uint8), cmap = 'Blues'), plt.axis('off'), plt.title('Masking images',fontsize=20), plt.show()
# Uniform addition of pixel values to images

eye_01x = (eye_01 * 0.5 + eye_02x * 0.2 + (96, 128, 160)).clip(0,255)

plt.imshow(eye_01x.astype(np.uint8), cmap = 'Blues'), plt.axis('off'), plt.title('Uniform addition',fontsize=20), plt.show()
# Mask creation by drawing in image

mask_01 = np.zeros_like(image_02[0:300,0:400])

cv2.rectangle(mask_01, (50, 50), (100, 200), (255, 255, 255), thickness=-1)

cv2.circle(mask_01, (200, 100), 50, (255, 255, 255), thickness=-1)

cv2.fillConvexPoly(mask_01, np.array([[330, 50], [300, 200], [360, 150]]), (255, 255, 255))

mask_01x = cv2.resize(mask_01,image_02.shape[1::-1])

plt.imshow(mask_01), plt.axis('off'), plt.title('Sample Mask',fontsize=20), plt.show()
# Bitwise and with the mask created

plt.figure(figsize=(10,10))

plt.imshow(cv2.bitwise_and(image_02,mask_01x)), plt.axis('off'), plt.title('Bitwise masking',fontsize=20), plt.show()
# Reading a new image for working with image channels

image_03 = cv2.imread(BASE+'1 (4956).jpg')

print(image_03.shape)



# Splitting the channels

plt.figure(figsize=(15,15))

b,g,r = cv2.split(image_03)

mask_03 = np.zeros(image_03.shape[:2], dtype = "uint8")

image_03x = cv2.merge((mask_03,g,r))

plt.subplot(221), plt.imshow(image_03[:,:,0], cmap= 'gray'), plt.axis('off'), plt.title('Red Channel',fontsize=20)

plt.subplot(222), plt.imshow(image_03[:,:,1], cmap= 'gray'), plt.axis('off'), plt.title('Green Channel',fontsize=20) 

plt.subplot(223), plt.imshow(image_03[:,:,2], cmap= 'gray'), plt.axis('off'), plt.title('Blue Channel',fontsize=20) 

plt.subplot(224), plt.imshow(image_03x), plt.axis('off'), plt.title('Channels Merged',fontsize=20)

plt.subplots_adjust(wspace=0, hspace=-0.25)

plt.show()
# Crop and Resize Images

height, width = image_03.shape[:2]

quarter_height, quarter_width = height / 4, width / 4

T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]]) 



plt.figure(figsize=(20,15))

plt.subplot(231), plt.imshow(image_03), plt.axis('off'), plt.title('Original Image',fontsize=20)

plt.subplot(232), plt.imshow(cv2.resize(image_03,(200,200))), plt.axis('off'), plt.title('Resized Image',fontsize=20)

plt.subplot(233), plt.imshow(image_03[470:610,420:610]), plt.axis('off'), plt.title('Cropped Image',fontsize=20)

plt.subplot(234), plt.imshow(cv2.warpAffine(image_03, T, (width,height)) ), plt.axis('off'), plt.title('Translated Image',fontsize=20)

plt.subplot(235), plt.imshow(cv2.rotate(image_03, cv2.ROTATE_90_CLOCKWISE)), plt.axis('off'), plt.title('Rotated Image',fontsize=20)

plt.subplot(236), plt.imshow(np.flip(image_03,(0, 1))), plt.axis('off'), plt.title('Flipped Image',fontsize=20)