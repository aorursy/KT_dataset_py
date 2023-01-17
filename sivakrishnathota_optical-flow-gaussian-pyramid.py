# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

from scipy import signal

from PIL import Image

import math

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img1=np.array(Image.open("/kaggle/input/mall-data-in-seq/seq_000009.jpg").convert('L'))

img2=np.array(Image.open("/kaggle/input/mall-data-in-seq/seq_000010.jpg").convert('L'))
print("img1 shape",img1.shape)

print("img2 shape",img2.shape)
plt.figure(figsize=(15,15))

plt.title('Frame 9')

plt.imshow(img1,cmap='gray')

plt.show()

plt.figure(figsize=(15,15))

plt.title('Frame 10')

plt.imshow(img2,cmap='gray')

plt.show()
def gaussian(sigma,x,y):

    a= 1/(np.sqrt(2*np.pi)*sigma)

    b=math.exp(-(x**2+y**2)/(2*(sigma**2)))

    c = a*b

    return a*b
def gaussian_kernal():

    G=np.zeros((5,5))

    for i in range(-2,3):

        for j in range(-2,3):

            G[i+1,j+1]=gaussian(1.5,i,j)

    return G
## Expand 

def Lucas_Kanade_Expand(image):

    w,h=image.shape

    newwidth=int(w*2)

    newheight=int(h*2)

    newimage=np.zeros((newwidth,newheight))

    newimage[::2,::2]=image

    G=gaussian_kernal()

    #print(G)

    for i in range(2,newimage.shape[0]-2,2):

        for j in range(2,newimage.shape[1]-2,2):

            newimage[i,j]=np.sum(newimage[i-2:i+3,j-2:j+3]*G)

    return newimage 

    

    
def Lucas_Kanade_Reduce(I1):

    w, h = I1.shape

    newWidth = int(w / 2)

    newHei = int(h / 2)

    G = gaussian_kernal()

    newImage = np.ones((newWidth, newHei))

    for i in range(2, I1.shape[0] - 2, 2):  # making image of half size by skiping alternate pixels

        for j in range(2, I1.shape[1] - 2, 2):

            newImage[int(i / 2), int(j / 2)] = np.sum(I1[i - 2:i + 3, j - 2:j + 3] * G)  # convolving with gaussian mask



    return newImage
def LK_Expand_Iterative(image,level):

    if level ==0:

        return image

    i=0

    while(i<level):

        image=Lucas_Kanade_Expand(image)

        i=i+1

    return image
def LK_Reduce_Iterative(Img,Level):

    if Level==0:#level 0 means current level i.e. no change

        return Img

    i=0

    while(i<Level):

        Img=Lucas_Kanade_Reduce(Img)

        i=i+1

    return Img
newimg=LK_Expand_Iterative(img1,1)

print("newimg shape ",newimg.shape)

plt.figure(figsize=(15,15))

plt.imshow(newimg,cmap='gray')
newim=LK_Reduce_Iterative(newimg,1)

print("Shape",newim.shape)

plt.figure(figsize=(15,15))

plt.imshow(newim,cmap='gray')
plt.hist(img1)
plt.hist(img2)