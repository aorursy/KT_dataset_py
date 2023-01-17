import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import cv2

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

from keras.models import Model



train = pd.read_csv('../input/digit-recognizer/train.csv')



images = pd.DataFrame()

def split_dataset(x,x_train):

    train_sub = train[train['label']==x].sample(4) #I will only be using 4 examples from each number

    x_train = x_train.append(train_sub,ignore_index=True)

    return x_train



for i in range (10):

    images = split_dataset(i,images)



print(images.head(4))



images = images.drop(['label'],axis = 1)

images = images.values.reshape([-1,28,28,1]) 
def display_one (a):

    plt.imshow(a,cmap = 'gray')

    plt.axis('off') 

    plt.show()

def display(images):

    y = 0

    for i in range (10):

        plt.subplot(141), plt.imshow(images[y],cmap='gray')

        plt.axis('off')

        plt.subplot(142), plt.imshow(images[y+1],cmap='gray')

        plt.axis('off')

        plt.subplot(143), plt.imshow(images[y+2],cmap='gray')

        plt.axis('off')

        plt.subplot(144), plt.imshow(images[y+3],cmap='gray')

        plt.axis('off')

        y+=4

        plt.show()



def display_change (images, func):

    y = 0

    for i in range (10):

        plt.subplot(181), plt.imshow(images[y],cmap='gray')

        plt.axis('off')

        plt.subplot(182), plt.imshow(images[y+1],cmap='gray')

        plt.axis('off')

        plt.subplot(183), plt.imshow(images[y+2],cmap='gray')

        plt.axis('off')

        plt.subplot(184), plt.imshow(images[y+3],cmap='gray')

        plt.axis('off')

        plt.subplot(185), plt.imshow(func(images[y]),cmap='gray')

        plt.axis('off')

        plt.subplot(186), plt.imshow(func(images[y+1]),cmap='gray')

        plt.axis('off')

        plt.subplot(187), plt.imshow(func(images[y+2]),cmap='gray')

        plt.axis('off')

        plt.subplot(188), plt.imshow(func(images[y+3]),cmap='gray')

        plt.axis('off')

        y+=4

        plt.show()

def size(img):

    img = array_to_img(img, scale = False)

    img = img.resize((100,100))

    img = img.convert(mode = 'RGB')

    img = img_to_array(img)

    return img.astype(np.float64)

    

resized = []

for i in images:

    resized.append(size(i))

    

display(resized)
resized = np.array(resized)

def normalized (img):

    return img/255.0



display_change(resized,normalized)
def averaging(img):

    return cv2.blur(img,(5,5))



display_change(resized,averaging)
def gaussian(img):

    return cv2.GaussianBlur(img,(5,5),0)



display_change(resized,gaussian)
def median(img):

    return cv2.medianBlur(np.float32(img),3)



display_change(resized, median)
def bilateral(img):

    return cv2.bilateralFilter(img.astype(np.uint8),9,75,75)



display_change(resized,bilateral)
kernel = np.ones((5,5),np.uint8)



def dilation(img):

    return cv2.dilate(img,kernel,iterations = 1)



display_change(resized,dilation)
def dilation_1(img):

    return cv2.dilate(img,kernel,iterations = 5)



display_change(resized,dilation_1)
kernel = np.ones((10,10),np.uint8)





display_change(resized,dilation)
kernel = np.ones((5,5), np.uint8) 



def erosion(img):

    return cv2.erode(img,kernel,iterations = 1)



display_change(resized,erosion)
def erode_1(img):

    return cv2.erode(img,kernel,iterations = 2)



display_change(resized,erode_1)
kernel = np.ones((10,10),np.uint8)





display_change(resized,erosion)
kernel = np.ones((5,5),np.uint8)



def closing(img):

    return erosion(dilation(img))



display_change(resized,closing)
def opening(img):

    return dilation(erosion(img))



display_change(resized,opening)
kernel_sharpening = np.array([[-2,-2,-2],

                             [-2,25,-2],

                             [-2,-2,-2]])



def sharpening(img):

    return cv2.filter2D(img,-1,kernel_sharpening)



display_change(resized,sharpening)