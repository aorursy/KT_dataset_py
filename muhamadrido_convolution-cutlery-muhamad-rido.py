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
image = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')
sharpening_img = np.array([[-1,-1,-1],[-1,9,-1],[-0,-2,-0]])
sharped_fix = cv2.filter2D(image, -1, sharpening_img)
plt.figure(figsize=(7,7))
plt.imshow(image)
plt.title('ORIGINAL IMAGE'), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(7,7))
plt.imshow(sharped_fix)
plt.title('SHARPENING IMAGE'), plt.xticks([]), plt.yticks([])
a = plt.imread('/kaggle/input/cutlery/cutlery.jpg')
a1 = plt.imread('/kaggle/input/cutlery/cutlery.jpg')
a = a.sum(axis=-1)
def blur(a):
    for x in range(1,a.shape[0]-1):
        for y in range(1,a.shape[1]-1):
            a[x,y] = (30*a[x,y] + a[x-1,y] + a[x+1,y] + a[x,y-1] + a[x,y+1])/8.0
    return a
%timeit blur(a)
plt.figure(figsize=(7,7))
plt.imshow(a1, cmap="gray")
plt.title('ORIGINAL IMAGE'), plt.xticks([]), plt.yticks([])
blur(a)
blur(a)
blur(a)
plt.figure(figsize=(7,7))
plt.imshow(a, cmap="gray")
plt.title('BLURING IMAGE'), plt.xticks([]), plt.yticks([])
img = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge_Canny = cv2.Canny(img_gray,120,220)
sobel_x = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=7)
sobel_y = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=7)
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
                   [1, 2,  1]])

app = filter2d(gray, kernel)
plt.subplot(2,2,4),plt.imshow(edge_Canny,cmap = 'gray')
plt.title('CANNY IMAGE'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobel_x,cmap = 'gray')
plt.title('SOBEL X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobel_y,cmap = 'gray')
plt.title('SOBEL Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,1),plt.imshow(app,cmap = 'gray')
plt.title('LAPLACIAN IMAGE'), plt.xticks([]), plt.yticks([])

plt.show()