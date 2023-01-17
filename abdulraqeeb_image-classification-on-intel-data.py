import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import random

import cv2
traindata="../input/intel-image-classification/seg_train/seg_train"

category=["buildings","forest","glacier","mountain","sea","street"]
for i in category:

    path=os.path.join(traindata,i)

    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array,cmap='gray')

        plt.show()

        break

testdata="../input/intel-image-classification/seg_test/seg_test"

category=["buildings","forest","glacier","mountain","sea","street"]
for i in category:

    path=os.path.join(testdata,i)

    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array,cmap='gray')

        plt.show()

        break

training_data=[]

def create_training_data():

    for i in category:

        path=os.path.join(traindata,i)

        class_num=category.index(i)

        for img in os.listdir(path):

            try:

                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_array=cv2.resize(img_array,(150,150),interpolation=cv2.INTER_AREA)

                training_data.append([new_array,class_num])

            except Exception as e:

                pass

create_training_data()
training_data
len(training_data)
random.shuffle(training_data)
for sample in training_data[:10]:

    print(sample[1])
testing_data=[]

def create_testing_data():

    for i in category:

        path=os.path.join(testdata,i)

        class_num=category.index(i)

        for img in os.listdir(path):

            try:

                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_array=cv2.resize(img_array,(150,150),interpolation=cv2.INTER_AREA)

                testing_data.append([new_array,class_num])

            except Exception as e:

                pass

create_testing_data()
testing_data
len(testing_data)
random.shuffle(testing_data)
for sample in testing_data[:10]:

    print(sample[1])
X=[]

y=[]
for features,label in training_data:

    X.append(features)

    y.append(label)
len(X)
len(y)
X=np.array(X).reshape(-1,150,150,1)

X.shape
X.shape[1:]
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
model=Sequential()

model.add(Conv2D(40,(3,3),input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(35,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(30,(3,3)))

model.add(Activation('relu'))

model.add(Conv2D(25,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(15,(3,3)))

model.add(Activation('relu'))

model.add(Conv2D(10,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(10,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(6))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

X = X/255.0

X
from tensorflow.keras.utils import to_categorical
y_ONEHOT=to_categorical(y)
model.fit(X,y_ONEHOT,validation_split=0.1,epochs=30)
y[0:15]
model.predict_classes(X[0:15])