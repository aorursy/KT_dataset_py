# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

maindir = '../input/gtsrb-german-traffic-sign'

# Any results you write to the current directory are saved as output.
import imageio

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report

from skimage import transform

from skimage import exposure

from skimage import io

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random

import os

import cv2

from PIL import Image
img=imageio.imread(maindir+"/meta/31.png")

plt.imshow(img)

plt.show()
data=[]

labels=[]



height = 30

width = 30

channels = 3

classes = 43

n_inputs = height * width*channels



for i in range(classes) :

    path = (maindir+"/train/{0}/").format(i)

    print(path)

    Class=os.listdir(path)

    for a in Class:

        try:

            image=cv2.imread(path+a)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((height, width))

            data.append(np.array(size_image))

            labels.append(i)

        except AttributeError:

            print(" ")

            

Cells=np.array(data)

labels=np.array(labels)



#Randomize the order of the input images

s=np.arange(Cells.shape[0])

np.random.seed(2)

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]
(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]

X_train = X_train.astype('float32')/255 

X_val = X_val.astype('float32')/255

(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]



#Using one hote encoding for the train and validation label

y_train = to_categorical(y_train, 43)

y_val = to_categorical(y_val, 43)
#np.save('xtrain', X_train)

#np.save('xval', X_val)

#np.save('ytrain', y_train)

#np.save('yval', y_val)
!ls -1
#X_train=np.load('xtrain.npy')

#X_val=np.load('xval.npy')

#y_train=np.load('ytrain.npy')

#y_val=np.load('yval.npy')
import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam
model = tf.keras.models.Sequential()

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(32, kernel_size=(2,2), input_shape=(X_train[0].shape)))

model.add(layers.MaxPooling2D((2,2),padding='same'))



#model.add(layers.Conv2D(64, kernel_size=(3,3)))

#model.add(layers.Conv2D(64, kernel_size=(2,2)))

#model.add(layers.MaxPooling2D((2,2),padding='same'))

#model.add(layers.Conv2D(128, kernel_size=(3,3)))

#model.add(layers.Conv2D(256, kernel_size=(3,3)))

#model.add(layers.MaxPooling2D((2,2),padding='same'))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

#model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(43, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))
import matplotlib

import matplotlib.pyplot as plt

acc=history.history['accuracy']

acc[0]=None

valacc=history.history['val_accuracy']

epochs = range(1,11)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, valacc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
y_test=pd.read_csv(maindir+"/Test.csv")

labels=y_test['Path'].values

y_test=y_test['ClassId'].values



data=[]



for f in labels:

    image=cv2.imread((maindir+'/test/')+f.replace('Test/', ''))

    image_from_array = Image.fromarray(image, 'RGB')

    size_image = image_from_array.resize((height, width))

    data.append(np.array(size_image))



X_test=np.array(data)

X_test = X_test.astype('float32')/255 

pred = model.predict_classes(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)
array=np.zeros(43)

for i in range(42):

    imgarray=(cv2.imread(maindir+('/meta/{}.png').format(i)))*1.

    imgarray=cv2.resize(imgarray,(30,30))

    imgarray=imgarray.reshape(1,imgarray.shape[0],imgarray.shape[0],3)



    array[i]=np.argmax(model.predict(imgarray))
gt = np.arange(43)
np.sum(array==gt)/43