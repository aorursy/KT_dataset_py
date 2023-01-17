# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2

import numpy as np 

from matplotlib import pyplot as plt
r = plt.imread('/kaggle/input/cutlery/cutlery.jpg')

r1 = plt.imread('/kaggle/input/cutlery/cutlery.jpg')

r = r.sum(axis=-1)
def blur(r) :

    for i in range(1,r.shape[0]-1):

        for j in range(1,r.shape[1]-1):

            r[i,j] = (15*r[i,j] + r[i-1,j] + r[i+1,j] + r[i,j-1] + r[i,j+1])/5.0

    return r
%timeit blur(r)
plt.figure(figsize=(8,7))

plt.imshow(r1, cmap="gray")

plt.title('Original'),plt.xticks([]), plt.yticks([])

blur(r)

blur(r)

blur(r)

plt.figure(figsize=(8,7))

plt.imshow(r, cmap="gray")

plt.title('has Blured'), plt.xticks([]), plt.yticks([])
img = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge_Canny = cv2.Canny(img_gray,100,200)
sobel_x = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=5)

sobel_y = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=5)
def filter2d(src, kernel):

    m, n = kernel.shape



    q = int((m-1)/2)

    r, s = src.shape[0], src.shape[1]



    fit = np.zeros((r, s))



    for y in range(q, r - q):

        for x in range(q, s - q):

           

            fit[y][x] = np.sum(src[y-q:y+q+1, x-q:x+q+1]*kernel)



    return fit



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.array([[1, 1,  1],

                   [1, -8, 1],

                   [1, 1,  1]])



fit = filter2d(gray, kernel)
plt.figure(figsize=(8,7))

plt.subplot(2,2,1),plt.imshow(edge_Canny,cmap = 'gray')

plt.title('Canny Image'), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(8,7))

plt.subplot(2,2,2),plt.imshow(sobel_x,cmap = 'gray')

plt.title('Sobel X Image'), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(8,7))

plt.subplot(2,2,3),plt.imshow(sobel_y,cmap = 'gray')

plt.title('Sobel Y Image'), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(8,7))

plt.subplot(2,2,4),plt.imshow(fit,cmap = 'gray')

plt.title('Laplacian Image'), plt.xticks([]), plt.yticks([])



plt.show()
import cv2

import numpy as np 



image = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')



kernel_sharpening = np.array([[-1,-1,-1],

                              [-1, 9,-1],

                              [-1,-1,-1]])



sharped = cv2.filter2D(image,-1,kernel_sharpening)
plt.figure(figsize=(7,8))

plt.imshow(image)

plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,8))

plt.imshow(sharped)

plt.title('This Is Sharpened'), plt.xticks([]),plt.yticks([])