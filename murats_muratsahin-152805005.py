# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
DATADIR = "../input/dataset"

CATEGORIES = ["fork","glass"]
for category in CATEGORIES:  # catallar ve bardaklar icin kategoriler

    path = os.path.join(DATADIR,category)  # dizin yolunu olusturuyoruz

    for img in os.listdir(path):  # her catal ve bardak resmi icin ilerlet for dongusu ile 

        img_array = cv2.imread(os.path.join(path,img))  # arraye ceviriyoruz

        plt.imshow(img_array, cmap='gray')  # color map, gray belirledik

        plt.show()  # gostersin

        break     

    break   
print(img_array)
print(img_array.shape)
IMG_HEIGHT = 30

IMG_WIDTH = 30
new_array = cv2.resize(img_array, (IMG_WIDTH,IMG_HEIGHT))
new_array.shape
plt.imshow(new_array, cmap='gray') 

plt.show()
training_data = []



def create_training_data():

    for category in CATEGORIES:  # catallar ve bardaklar

        path = os.path.join(DATADIR,category)  # resimlerin yolunu olustur

        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):  # her catal ve bardak icin devam et ilerlet,

            try:

                img_array = cv2.imread(os.path.join(path,img))  # array'e donustur.

                new_array = cv2.resize(img_array, (IMG_WIDTH,IMG_HEIGHT))

                training_data.append([new_array, class_num])

            except Exception as e:

                pass
create_training_data()
print(len(training_data))
import random

random.shuffle(training_data)
for sample in training_data[:10]:

    print(sample[1])
X = []

y = []



for features,label in training_data:

    X.append(features)

    y.append(label)



print(X[0].reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3))



X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)
plt.imshow(X[1])
import pickle



pickle_out = open("X.pickle","wb")

pickle.dump(X, pickle_out)

pickle_out.close()



pickle_out = open("y.pickle","wb")

pickle.dump(y, pickle_out)

pickle_out.close()
pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

Y = pickle.load(pickle_in)
X.shape
Y[0:10]
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle
pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

y = pickle.load(pickle_in)

X = X/255
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 3 boyutu tek boyuta indirgiyor flatten ile, düzlüyoruz.

model.add(Dense(64))

model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
history = model.fit(X, y, batch_size=2, epochs=6, validation_split=0.2)
results = model.evaluate(X, y)

results
history_dict = history.history

history_dict.keys()

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()  

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.add(Dropout(0.0001))
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(X, y, batch_size=2, epochs=10, validation_split=0.2)
results = model.evaluate(X, y)

results
flipped_img = np.fliplr(new_array)

plt.imshow(flipped_img)

plt.show()
for j in range(IMG_WIDTH):

    for i in range(IMG_HEIGHT):

        if (i < IMG_HEIGHT-1000):

          new_array[j][i] = img_array[j][i+1000]



plt.imshow(new_array)

plt.show()
noise = np.random.randint(5, size = (164, 278, 4), dtype = 'uint8')



for i in range(IMG_WIDTH):

    for j in range(IMG_HEIGHT):

        for k in range(2):

            if (new_array[i][j][k] != 255):

                new_array[i][j][k] += noise[i][j][k]

plt.imshow(new_array)

plt.show()
from keras import regularizers

from keras import layers

from keras import models



model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))

model.add(Dense(64, activation='relu'))

model.add(Dense(1))



#model.add(layers.Dense(16, 

#           kernel_regularizer = regularizers.12(0.001), 

#           activation=relu,input_shape =1000,))



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model = Sequential()

model.add(Conv2D(256, (4, 4), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(256, (4,4)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())  #this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer='RMSprop', #optimizers.RMSprop(lr=1e-4),

              metrics=['accuracy'])
history = model.fit(X, 

                    y,

                    batch_size=2,

                    epochs=5,

                    validation_split=0.2)
history_dict = history.history

history_dict.keys()

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()