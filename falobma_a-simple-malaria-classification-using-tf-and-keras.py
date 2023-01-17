# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from PIL import Image

import numpy as np

import os

import cv2

import tensorflow as tf

from tensorflow import keras

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
#Getting the image from files

# using cv2 and put it in arrays for the data and create the label based on the category

data=[]

labels=[]

Parasitized=os.listdir("../input/cell_images/cell_images/Parasitized/")

for parasite in Parasitized:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Parasitized/"+parasite)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        data.append(np.array(size_image))

        labels.append(0)

    except AttributeError:

        print("")



Uninfected=os.listdir("../input/cell_images/cell_images/Uninfected/")

for uninfect in Uninfected:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Uninfected/"+uninfect)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        data.append(np.array(size_image))

        labels.append(1)

    except AttributeError:

        print("")

    
df = np.array(data)

labels = np.array(labels)

(X_train, X_test) = df[(int)(0.1*len(df)):],df[:(int)(0.1*len(df))]

(y_train, y_test) = labels[(int)(0.1*len(labels)):],labels[:(int)(0.1*len(labels))]
s=np.arange(X_train.shape[0])

np.random.shuffle(s)

X_train=X_train[s]

y_train=y_train[s]

X_train = X_train/255.0
model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()
model.compile(optimizer="adam",

              loss="sparse_categorical_crossentropy", 

             metrics=["accuracy"])
model.fit(X_train,y_train, epochs=10)
X_loss, accuracy = model.evaluate(X_test,y_test)



print(accuracy)