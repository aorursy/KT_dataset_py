# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import os

import cv2

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = '../input/flowers/flowers'

os.listdir(path)
size = (64,64)



imgs_label = []

train_labels = []

imgs_train = []





for f in os.listdir(path):

    for file in os.listdir(os.path.join(path,f)):

        if file.endswith("jpg"):

            imgs_label.append(os.path.join(path,f,file))

            train_labels.append(f)

            img = cv2.imread(os.path.join(path,f,file))

            img = cv2.resize(img,size)

            imgs_train.append(img)
train = np.array(imgs_train, dtype='float32')
train = train/255
labels = pd.get_dummies(train_labels)

labels =  labels.values.argmax(1)
labels
print(pd.unique(labels))

print(pd.unique(train_labels))
ul = list(zip(train, labels))



random.shuffle(ul)

train,labels = zip(*ul)



# Convert the shuffled list to numpy array type



train = np.array(train)

labels = np.array(labels)
model1 = keras.Sequential([

    keras.layers.Conv2D(filters=32,kernel_size=(3,3), padding='Same',activation='relu', input_shape=(64,64,3)),

    keras.layers.MaxPooling2D(pool_size=(2,2)),

    keras.layers.Flatten(input_shape=(64,64,3)),

    keras.layers.Dense(128),

    keras.layers.Dense(5, activation=tf.nn.softmax)

])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.30, shuffle='False')
X_train.shape
model1.compile(optimizer=tf.train.AdamOptimizer(), 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model1.fit(X_train,y_train, epochs=15,batch_size=128,validation_data=(X_test, y_test))
pred = model1.predict_classes(X_test[:10])



for i in range(len(pred)):

    print(pred[i],'==>',y_test[i])
