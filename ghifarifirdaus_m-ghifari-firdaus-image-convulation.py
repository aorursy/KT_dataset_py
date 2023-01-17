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

from matplotlib import pyplot as plt
c = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')

c_ori = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

plt.title('Original')

plt.imshow(c_ori)

plt.show()
cl = plt.imread('/kaggle/input/cutlery/cutlery.jpg')

cl2 = plt.imread('/kaggle/input/cutlery/cutlery.jpg')

cl = cl.sum(axis=-1)



def blur(cl):

    for i in range(1, cl.shape[0]-1):

        for j in range(1, cl.shape[1]-1):

            cl[i,j] = (10*cl[i,j] + cl[i-1,j] + cl[i+1,j] + cl[i,j+1] + cl[i,j-1])/3

    return cl
%timeit blur(cl)
plt.figure(figsize=(7,7))

plt.imshow(cl2, cmap="gray")

plt.title('Original'), plt.xticks([]), plt.yticks([])

blur(cl)

plt.figure(figsize=(7,7))

plt.imshow(cl, cmap="gray")

plt.title('Blurred'), plt.xticks([]), plt.yticks([])
cultery = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')

cultery = cv2.cvtColor(cultery, cv2.COLOR_BGR2RGB)

kernel_sharp = np.array([[-1, -1, -1],

                         [-1,  9, -1],

                         [-1, -1, -1]])

sharp = cv2.filter2D(cultery, -1, kernel_sharp)



plt.figure(figsize=(7,7))

plt.imshow(cultery)

plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(7,7))

plt.imshow(sharp)

plt.title('Sharp Mode'), plt.xticks([]), plt.yticks([])
c = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')

cgray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)



canny = cv2.Canny(cgray,100,200)



sobelx = cv2.Sobel(cgray, cv2.CV_64F,1,0,ksize=5)

sobely = cv2.Sobel(cgray, cv2.CV_64F,0,1,ksize=5)
def f2d(src, kernel):

    m,n = kernel.shape

    d = int((m-1)/2)

    h,w = src.shape[0], src.shape[1]

    

    dst = np.zeros((h,w))

    

    for y in range(d, h-d):

        for x in range(d, w-d):

            dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)

        return dst

    

cugray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

kernel = np.array([[1,  1, 1],

                  [ 1, -8, 1],

                  [ 1,  1, 1]])



dst = f2d(cugray, kernel)



plt.subplot(2,2,1), plt.imshow(canny, cmap='gray')

plt.title('Canny'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2), plt.imshow(sobelx, cmap='gray')

plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3), plt.imshow(sobely, cmap='gray')

plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4), plt.imshow(dst, cmap='gray')

plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.show()
plt.imshow(cugray,cmap = 'gray')

plt.title('Original Gray'), plt.xticks([]), plt.yticks([])



plt.show()