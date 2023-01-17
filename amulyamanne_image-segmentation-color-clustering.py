# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
image1='/kaggle/input/original-image/image1.jpg'

image=cv2.imread(image1)

plt.axis('off')

plt.imshow(image)

plt.title('Original image')

plt.show()
img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.axis('off')

plt.imshow(img)

plt.title('Original Image')

plt.show()
img.shape
vectorized = np.float32(img.reshape((-1,3)))

vectorized.shape
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
K = 3

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(2,3,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
K = 4

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(2,3,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
K = 5

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(2,3,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
K = 6

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(2,3,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
K = 7

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(2,3,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
K = 8

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(2,3,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()
K = 9

attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(2,3,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()