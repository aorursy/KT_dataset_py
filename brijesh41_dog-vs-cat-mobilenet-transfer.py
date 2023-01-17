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
from PIL import Image

from zipfile import ZipFile

from tqdm import tqdm

from cv2 import cv2

from keras.layers import Conv2D,Dense,Flatten

from keras.models import Sequential

import keras

import tensorflow
img_width = 150

img_height = 150

TRAIN_DIR = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train'

TEST_DIR = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test'

train_images_dogs_cats = []

test_images_dogs_cats= []
labels = []

for img in tqdm(os.listdir(TRAIN_DIR)):

    

    try:

        

        imgr = cv2.imread(os.path.join(TRAIN_DIR,img))

        train_images_dogs_cats.append(cv2.resize(imgr,(150,150),interpolation=cv2.INTER_CUBIC))

        

        if 'dog' in img:

                labels.append(1)

        else:

                labels.append(0)

    except Exception as e:

        

        print("bad image")

        

       


from sklearn.model_selection import train_test_split
print(len(train_images_dogs_cats))

print(len(labels))

print(train_images_dogs_cats[1].shape)

x_train,y_train,x_test,y_test = train_test_split(train_images_dogs_cats,labels, test_size = 0.2)
print(len(x_train))

print(len(y_train))

print(len(x_test))
mobilenet_path = '../input/mobilenet/mobilenet_1_0_224_tf_no_top.h5'
model = Sequential()

model.add(keras.applications.mobilenet.MobileNet(input_shape=(150,150,3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights=mobilenet_path, input_tensor=None, pooling=None, classes=1000))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
model.fit(np.array(x_train),np.array(x_test),epochs=3,validation_data=(np.array(y_train),np.array(y_test)))
test_data_labels = []

test_data_image = []
for img in tqdm(os.listdir('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test')):

    try:

        imgr = cv2.imread(os.path.join(TEST_DIR,img))

        test_data_image.append(cv2.resize(imgr,(150,150),interpolation=cv2.INTER_CUBIC))

    except Exception as e:

        print(',')

        

        

    

    
print(test_data_image[12499].shape)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

count = 0

for img in tqdm(os.listdir('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test')):

    count = count+1

    if count>10:

        break;

    try:

        imgr = cv2.imread(os.path.join(TEST_DIR,img))

        #test_data_image.append(cv2.resize(imgr,(150,150),interpolation=cv2.INTER_CUBIC))

        imgplot = plt.imshow(imgr)

        plt.show()

    except Exception as e:

        print(',')

        

test_data_labels = model.predict(np.array(test_data_image))
print(test_data_labels[1])

if np.argmax(test_data_labels[9]) == 1: 

        str_predicted='Dog'

else: 

        str_predicted='Cat'

print(str_predicted)
solution = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
solution.label = test_data_labels
solution.to_csv("dogsVScats.csv", index = False)