import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Sequential

from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers

from keras.layers import Conv2D, Flatten, MaxPooling2D, Input

import cv2

import os
labels = os.listdir('../input/natural-images/natural_images')

labels
from IPython.display import display, Image



for label in labels:

    path = '../input/natural-images/natural_images/{}/'.format(label)

    img_data = os.listdir(path)

    

    k=0

    for image_data in img_data:

        if k < 3:

            display(Image(path+image_data))

        k += 1
x = []

y = []



for label in labels:

    path = '../input/natural-images/natural_images/{}/'.format(label)

    img_data = os.listdir(path)

    

    for image in img_data:

        a=cv2.imread(path+image)

        a = cv2.resize(a,(64,64))

        x.append(np.array(a.astype('float32'))/255)

        y.append(label)
plt.imshow(x[0])
x = np.array(x)

x.shape
# x = x.reshape(-1,64,64,3)
x[0].shape
y_d = []

for i in y:

    if i == 'airplane':

        y_d.append(0)

    elif i == 'car':

        y_d.append(1)

    elif i == 'cat':

        y_d.append(2)

    elif i == 'dog':

        y_d.append(3)

    elif i == 'flower':

        y_d.append(4)

    elif i == 'fruit':

        y_d.append(5)

    elif i == 'human':

        y_d.append(6)

    else:

        y_d.append(7)
y = y_d
plt.imshow(x[5000])

y[5000]
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.27)
# xtrain = xtrain.reshape(-1,64,64,4)

# xtrain.shape
from keras import models, layers

model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(64,64,3)))

model.add(layers.MaxPool2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(layers.MaxPool2D(pool_size=(2, 2)))

# model.add(layers.Dropout(rate=0.25))

model.add(layers.Flatten())

# model.add(layers.Dense(256, activation='relu'))

# model.add(layers.Dropout(rate=0.5))

model.add(layers.Dense(8, activation='softmax'))
model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
ytrain = np.array(ytrain)
ytrain
model.summary()
model.fit(xtrain,ytrain,batch_size=(256),epochs=25)
# plt.figure(figsize=(6,6))

# plt.plot(history.history['accuracy'])

# plt.plot(history.history['loss'])

# plt.show()
pred = model.predict(xtest)
diff = []

for i in pred:

    diff.append(np.argmax(i))
from sklearn.metrics import accuracy_score
accuracy_score(diff,ytest)