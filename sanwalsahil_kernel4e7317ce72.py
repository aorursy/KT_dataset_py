# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import PIL

import tensorflow as tf

import seaborn as sns

import pickle

from sklearn.utils import shuffle

import cv2
# importing pickle data

with open('../input/traffic-signs-preprocessed/train.pickle',mode='rb') as f:

    train = pickle.load(f)

with open('../input/traffic-signs-preprocessed/valid.pickle',mode='rb') as f:

    valid = pickle.load(f)

with open('../input/traffic-signs-preprocessed/test.pickle',mode='rb') as f:

    test = pickle.load(f)
x_train,y_train = train['features'],train['labels']

x_valid,y_valid = valid['features'],valid['labels']

x_test,y_test = test['features'],test['labels']
len(x_train)
train.keys()
x_train.shape
y_train.shape
i = 4500

plt.imshow(x_train[i])

y_train[i]
#shuffling the dataset

x_train,y_train = shuffle(x_train,y_train)
def display(a, b, title1 = "Original", title2 = "Edited"):

    plt.subplot(121), plt.imshow(a), plt.title(title1)

    plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(b), plt.title(title2)

    plt.xticks([]), plt.yticks([])

    plt.show()
#image data preprocessing



def processing(data):

    i = 0

    for image in data:

        i = i+1

        print('####################### '+str(i)+' #####################')

        org = image

        #setting dimensions of the resize

        height=32

        width=32

        dim = (width,height)

        

        image = cv2.resize(image,dim,interpolation=cv2.INTER_LINEAR)

        

        #----------- NOISE REMOVAL---------

        #gaussian blur

        image = cv2.GaussianBlur(image,(5,5),0)

        

        #display(org,image,'orginal','modified')

        

        # segmentation

        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        

        #display(org,image,'orginal','modified')

        ret,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        

        #display(org,image,'orginal','modified')

        

        # further noise removal

        kernel = np.ones((3,3),np.uint8)

        image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel,iterations=2)

        #display(org,image,'orginal','modified')

        

        # sure background area

        sure_bg = cv2.dilate(image,kernel,iterations=3)

        #print('---------- sure background ----------')

        #display(org,sure_bg,'orginal','modified')

        

        # finding sure foreground area

        dist_transform = cv2.distanceTransform(image,cv2.DIST_L2, 5)

        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        #print('-------------- sure_fg -------------')

        #display(org,sure_bg,'orginal','modified')

        #finding unknown region

        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        #print('-------------- unknown -------------')

        #display(org,unknown,'orginal','modified')

        

        #marker labeling

        ret,image = cv2.connectedComponents(sure_fg)

        #print('-------------- markers -------------')

        #display(org,image,'orginal','modified')

        

        # Add one to all labels so that sure background is not 0, but 1

        image = image + 1



        # Now, mark the region of unknown with zero

        image[unknown == 255] = 0

        

        image = cv2.watershed(org, image)

        org[image == -1] = [255, 0, 0]



        # Displaying markers on the image

        #display(org, image, 'Original', 'Marked')

        

        

        

        

    

        
processing(x_train)
plt.imshow(x_train[0])
processing(x_valid)
processing(x_test)

# comverting to gray

x_train_gray = np.sum(x_train/3,axis=3,keepdims = True)

x_valid_gray = np.sum(x_valid/3,axis=3,keepdims = True)

x_test_gray = np.sum(x_test/3,axis=3,keepdims = True)
#normalisation

x_train_gray = (x_train_gray)/255

x_valid_gray = (x_valid_gray)/255

x_test_gray = (x_test_gray)/255
plt.imshow(x_train_gray[0].squeeze(),cmap='gray')
from tensorflow.keras import datasets, layers, models



LeNet = models.Sequential()



LeNet.add(layers.Conv2D(8, (5,5), activation = 'relu', input_shape = (32,32,1)))

LeNet.add(layers.AveragePooling2D())





LeNet.add(layers.Conv2D(16, (5,5), activation = 'relu'))

LeNet.add(layers.AveragePooling2D())



LeNet.add(layers.Conv2D(32, (3,3), activation = 'relu'))

LeNet.add(layers.AveragePooling2D())



LeNet.add(layers.Flatten())



LeNet.add(layers.Dense(120, activation = 'relu'))



LeNet.add(layers.Dense(84, activation = 'relu'))



LeNet.add(layers.Dense(43, activation = 'softmax'))

LeNet.summary()

LeNet.compile(optimizer='Adam', loss = 'sparse_categorical_crossentropy',

             metrics=['accuracy'])
history = LeNet.fit(x_train_gray,y_train,epochs = 30,

                    validation_data = (x_valid_gray,y_valid))
LeNet.save('final1.h5')
from IPython.display import FileLink

FileLink('final1.h5')