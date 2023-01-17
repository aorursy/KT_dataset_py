# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/annotations"))

print(os.listdir("../input/images"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical

from keras.layers import Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import pickle

import cv2

import pandas as pd

import random

import bs4

import xml

np.random.seed(0)
data=[]

x_data=[]

y_data=[]

kategori=[]

total_kategori = 120
ctr = 0

for filename in os.listdir("../input/annotations/Annotation"):

    kategorinya = filename[10:]

    kategori.append(kategorinya)

    for gambarnya in os.listdir("../input/annotations/Annotation/%s" %(filename)):

        filenya = open("../input/annotations/Annotation/"+filename+"/"+gambarnya, "r")

        koor = filenya.read()

        soup = bs4.BeautifulSoup(koor,"xml")

        xmin = soup.annotation.object.xmin.text

        ymin = soup.annotation.object.ymin.text

        xmax = soup.annotation.object.xmax.text

        ymax = soup.annotation.object.ymax.text

        filenya.close()

        img = cv2.imread("../input/images/Images/"+filename+"/"+gambarnya+".jpg")

        img_hasil = img[int(ymin):int(ymax),int(xmin):int(xmax),:]

        img_hasil = cv2.resize(img_hasil,(64,64))

        data.append([img_hasil, ctr])      

    ctr+=1
def preprocess(img):

  img = cv2.GaussianBlur(img,(5,5),0)

  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  img[:,:,2] = cv2.equalizeHist(img[:,:,2]) 

  img = img/255

  return img
ukuran = 64

chanel = 3
data = np.array(data)

random.shuffle(data)

for gambar, label in data:

  x_data.append(preprocess(gambar))

  y_data.append(label)

x = np.array(x_data)

y = np.array(y_data)



y = to_categorical(y,total_kategori)

print(x.shape)

print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25 , random_state= 42)

print(x_train.shape)

print(x_val.shape)

print(x_test.shape)
x_train = x_train.reshape(x_train.shape[0], ukuran, ukuran, chanel)

x_val = x_val.reshape(x_val.shape[0], ukuran, ukuran, chanel)

x_test = x_test.reshape(x_test.shape[0], ukuran, ukuran, chanel)



print(x_train.shape)
datagen = ImageDataGenerator(width_shift_range = 0.1,

                            height_shift_range = 0.1,

                            shear_range = 0.1,

                            rotation_range = 20,

                            horizontal_flip=True,

                            zoom_range = [0.9, 1.1]

                            )



datagen.fit(x_train)
batches = datagen.flow(x_train, y_train, batch_size=30)

x_batch, y_batch = next(batches)

fig, axs = plt.subplots(1,15, figsize=(20,5))

fig.tight_layout()



for i in range(15):

  if chanel == 1:

    axs[i].imshow(x_batch[i].reshape(ukuran,ukuran), cmap="gray")

  else:

    axs[i].imshow(x_batch[i].reshape(ukuran,ukuran,chanel))

  axs[i].axis("off")
from keras.layers import BatchNormalization

from keras import optimizers

def leNet_model():

  model = Sequential()

  model.add(Conv2D(150,(5,5), input_shape=(ukuran,ukuran,chanel), activation="relu"))

  model.add(Conv2D(150,(5,5), activation="relu"))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(BatchNormalization())

  model.add(Conv2D(120,(5,5), activation="relu",padding="same"))

  model.add(Conv2D(120,(3,3), activation="relu",padding="same"))

  model.add(Conv2D(90,(3,3),  activation="relu"))

  model.add(Conv2D(90,(3,3),  activation="relu"))

  #model.add(Dropout(0.5))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(BatchNormalization())

  

  model.add(Conv2D(60,(3,3),  activation="relu",padding="same"))

  model.add(Conv2D(60,(3,3),  activation="relu",padding="same"))

  #model.add(Dropout(0.5))

  model.add(Conv2D(60,(3,3),  activation="relu",padding="same"))

  model.add(Conv2D(30,(3,3),  activation="relu",padding="same"))

  model.add(Conv2D(30,(3,3),  activation="relu"))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(BatchNormalization())

  model.add(Conv2D(15,(2,2),  activation="relu",padding="same"))

  model.add(Conv2D(15,(2,2),  activation="relu"))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(BatchNormalization())

  

  model.add(Flatten())

  #model.add(Dense(500, activation="relu"))

  #model.add(Dropout(0.5))

  model.add(Dense(250, activation="relu"))

  model.add(Dropout(0.5))

  model.add(Dense(total_kategori, activation="softmax"))

  model.compile(Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])

  return model



model = leNet_model()

print(model.summary())
h = model.fit_generator(datagen.flow(x_train,y_train, batch_size=50),

                        steps_per_epoch=2000, epochs=20, 

                        validation_data=(x_val,y_val), shuffle=1, verbose=1)
plt.plot(h.history["loss"])

plt.plot(h.history["val_loss"])
h = model.fit_generator(datagen.flow(x_train,y_train, batch_size=50),

                        steps_per_epoch=2000, epochs=20, 

                        validation_data=(x_val,y_val), shuffle=1, verbose=1)
plt.plot(h.history["loss"])

plt.plot(h.history["val_loss"])
score = model.evaluate(x_test, y_test)

print(score[1])