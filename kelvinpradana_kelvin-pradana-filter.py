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
# get imagefile 
image = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')

# create the sharpen kernel
sharpen_kernel = np.array([[0,-1,0], 
                           [-1, 7,-1],
                           [0,-1,0]])
#apply the sharpen kernel
sharpened = cv2.filter2D(image, -1, sharpen_kernel)

plt.figure(figsize=(8,8))
plt.imshow(image)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(8,8))
plt.imshow(sharpened)
plt.title('Sharpen Image'), plt.xticks([]), plt.yticks([])
i = plt.imread('/kaggle/input/cutlery/cutlery.jpg')
ig = plt.imread('/kaggle/input/cutlery/cutlery.jpg')
i = i.sum(axis=-1)
def blur(i):
    for x in range(1,i.shape[0]-1):
        for y in range(1,i.shape[1]-1):
            i[x,y] = (10*i[x,y] + i[x-1,y] + i[x+1,y] + i[x,y-1] + i[x,y+1])/5.0
    return i
%timeit blur(i)
plt.figure(figsize=(8,6))
plt.imshow(ig, cmap="gray")
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
blur(i)
blur(i)
blur(i)
plt.figure(figsize=(8,6))
plt.imshow(i, cmap="gray")
plt.title('Blur Image'), plt.xticks([]), plt.yticks([])
img = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge_Canny = cv2.Canny(img_gray,100,200)
sobel_x = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=5)
sobel_y = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=5)
def filter2d(src, kernel):
    m, n = kernel.shape

    q = int((m-1)/2)
    r, s = src.shape[0], src.shape[1]

    app = np.zeros((r, s))

    for y in range(q, r - q):
        for x in range(q, s - q):
           
            app[y][x] = np.sum(src[y-q:y+q+1, x-q:x+q+1]*kernel)

    return app

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.array([[1, 1,  1],
                   [1, -8, 1],
                   [1, 1,  1]])

app = filter2d(gray, kernel)
plt.figure(figsize=(8,8))
plt.subplot(2,2,1),plt.imshow(edge_Canny,cmap = 'gray')
plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(8,8))
plt.subplot(2,2,2),plt.imshow(sobel_x,cmap = 'gray')
plt.title('Sobel X Image'), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(8,8))
plt.subplot(2,2,1),plt.imshow(sobel_y,cmap = 'gray')
plt.title('Sobel Y Image'), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(8,8))
plt.subplot(2,2,4),plt.imshow(app,cmap = 'gray')
plt.title('Laplacian Image'), plt.xticks([]), plt.yticks([])

plt.show()