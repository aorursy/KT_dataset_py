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
z = plt.imread('/kaggle/input/cutlery/cutlery.jpg')
r1 = plt.imread('/kaggle/input/cutlery/cutlery.jpg')
z = z.sum(axis=-1)
def blur(z):
    for i in range(1,z.shape[0]-1):
        for j in range(1,z.shape[1]-1):
            z[i,j] = (30*z[i,j] + z[i-1,j] + z[i+1,j] + z[i,j-1] + z[i,j+1])/8.0
    return z
%timeit blur(z)
plt.figure(figsize=(7,7))
plt.imshow(r1, cmap="gray")
plt.title('Ori'), plt.xticks([]), plt.yticks([])
blur(z)
blur(z)
blur(z)
plt.figure(figsize=(7,7))
plt.imshow(z, cmap="gray")
plt.title('Blur'), plt.xticks([]), plt.yticks([])
import cv2
import numpy as np
# Reading in and displaying our image
image = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')
# Create our shapening kernel, it must equal to one eventually
sharpening = np.array([[0,-1,0], 
                              [-1, 7,-1],
                              [0,-1,0]])
# applying the sharpening kernel to the input image & displaying it.
sharpened = cv2.filter2D(image, -1, sharpening)
plt.figure(figsize=(7,7))
plt.imshow(image)
plt.title('Gambar Original'), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(7,7))
plt.imshow(sharpened)
plt.title(' Gambar Sharpen'), plt.xticks([]), plt.yticks([])
img = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edgeCanny = cv2.Canny(img_gray,100,200)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=5)
def filter2d(src, kernel):
    m, n = kernel.shape

    d = int((m-1)/2)
    h, w = src.shape[0], src.shape[1]

    dst = np.zeros((h, w))

    for y in range(d, h - d):
        for x in range(d, w - d):
           
            dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)

    return dst

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.array([[1, 1,  1],
                   [1, -8, 1],
                   [1, 2,  1]])

dst = filter2d(gray, kernel)
plt.subplot(2,2,1),plt.imshow(edgeCanny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(dst,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
plt.imshow(gray,cmap = 'gray')
plt.title('Gambar Original'), plt.xticks([]), plt.yticks([])

plt.show()