import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import matplotlib.pyplot as plt

import matplotlib.image as img

from PIL import Image

data=[]

labels=[]

datadir=os.listdir('../input/wangdataset/Images/')

for b in datadir:

    try:

        image = cv2.imread('../input/wangdataset/Images/'+b)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((200,200))

        data.append(np.array(size_image))

        labels.append(b)

    except AttributeError:

        print("")

data=np.array(data)

data.shape
labels[5]
target=[]

for i in range(1000):

    if len(labels[i])==6:

        target.append(0)

    elif len(labels[i])==7:

        target.append(labels[i][0])

    elif len(labels[i])==5:

        target.append(0)

target1=np.array(target).astype(int)
target1.shape
import tensorflow as tf

import tensorflow.keras as keras

from keras.utils import to_categorical

no_classes=6

final_labels=to_categorical(target1)

final_labels.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(data,final_labels)
from keras.models import Sequential

from keras.layers import  Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(200,200,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dense(10,activation="softmax"))

model.summary()
np.array(X_train).shape
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size=50

epochs=40

predict=model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs)
accuracy = model.evaluate(X_test, y_test, verbose=1)

print('Test_Accuracy:-', accuracy[1])