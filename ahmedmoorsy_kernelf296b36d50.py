import numpy as np
from imutils import paths
import cv2
import sys
import keras
import os
import pandas as pd
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt 
from PIL import Image
import matplotlib.image as mpimg
%matplotlib inline
!ls ../input/data-release
imagePaths_train = sorted(list(paths.list_images("../input/data-release/train")))
imagePaths_test = sorted(list(paths.list_images("../input/data-release/test")))
print (len(imagePaths_train), len(imagePaths_test))
plt.figure(figsize=(12,10))
x, y = 10, 4
for i in range(40):  
    plt.subplot(y, x, i+1)
    img=mpimg.imread(imagePaths_train[i])
    plt.imshow(img,interpolation='nearest')
plt.show()
num_classes = 11
image_width = 118
image_height = 128 * 2
image_size = (image_width, image_height)
channels = 1
spectrogram = pd.read_csv("../input/data-release/train_labels.csv")
spectrogram.head()
images = []

for idd in spectrogram["id"]:
    c_imgPath = "../input/data-release/train/" + str(idd) + "_c.png"
    v_imgPath = "../input/data-release/train/" + str(idd) + "_v.png"
    c_image = cv2.imread(c_imgPath, cv2.IMREAD_GRAYSCALE)
    v_imgPath = cv2.imread(v_imgPath, cv2.IMREAD_GRAYSCALE)
    image = np.concatenate((c_image, v_imgPath))
    images.append(image)

images = np.array(images)
labels = np.array(spectrogram["appliance"])
len(images), len(labels),images.shape
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
len(x_train), len(x_test)

x_train = x_train.reshape(x_train.shape[0], image_width, image_height, 1)
x_test = x_test.reshape(x_test.shape[0], image_width, image_height, 1)
input_shape = (image_width, image_height, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from keras.models import model_from_json

import tensorflow as tf
    
def build_model(num_classes):
#we will use a sequential model for training 
    model = Sequential()

    #CONV 3x3x32 => RELU => NORMALIZATION => MAX POOL 3x3 block
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(3, 3)))

    #CONV 3x3x64 => RELU => NORMALIZATION => MAX POOL 2x2 block
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2, 2)))

    #CONV 3x3x128 => RELU => NORMALIZATION => MAX POOL 2x2 block
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2, 2)))

    #FLATTEN => DENSE 1024 => RELU => NORMALIZATION block
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    #final DENSE => SOFTMAX block for multi-label classification
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    #using categorical_crossentropy loss function with adam optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

from keras.callbacks import ModelCheckpoint
filepath='weights.best.hdf5'
model = build_model(num_classes)

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint]

history = model.fit( x_train, y_train,
          epochs=100,
          verbose=1,
          callbacks=callbacks_list,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])

