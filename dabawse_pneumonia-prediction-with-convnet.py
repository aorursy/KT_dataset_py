import cv2

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from pathlib import Path

from random import shuffle

from keras.regularizers import l1, l2

from keras.utils import to_categorical

from keras.optimizers import Adam, SGD

from keras.models import Sequential, Model

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Input, SeparableConv2D
data = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')

train_path = data / 'train'

test_path = data / 'test'

val_path = data / 'val'
train, test, val = [], [], []

X_train, y_train, X_test, y_test, X_val, y_val = [], [], [], [], [], []
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 20))

subplots = [ax1, ax2, ax3, ax4, ax5]



j = 0

for i in (train_path / 'NORMAL').glob('*.jpeg'):

    if j < 5:

        subplots[j].imshow(cv2.imread(str(i), 0))

        subplots[j].axhline(y=0.5, color='r')

    j += 1



plt.show()
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 20))

subplots = [ax1, ax2, ax3, ax4, ax5]



j = 0

for i in (train_path / 'PNEUMONIA').glob('*.jpeg'):

    if j < 5:

        subplots[j].imshow(cv2.imread(str(i), 0))

        subplots[j].axhline(y=0.5, color='r')

    j += 1



plt.show()
i = 0



for h in [(train_path / 'NORMAL').glob('*.jpeg'), (train_path / 'PNEUMONIA').glob('*.jpeg')]:

    for j in h:

        i += 1

        

        img = cv2.imread(str(j), 0)

        train.append(cv2.resize(img, (59536, 1))[0])

        train.append(str(j))
train = np.array(train).reshape(-1, 2)

train = np.array(pd.DataFrame(train).sample(frac=1).reset_index(drop=True))
for k in train:

    if k[1][58] == 'N':

        y_train.append(0)

    else:

        y_train.append(1)

    X_train.append(k[0])

    

X_train = pd.DataFrame(np.array(X_train).T)
i = 0



for h in [(test_path / 'NORMAL').glob('*.jpeg'), (test_path / 'PNEUMONIA').glob('*.jpeg')]:

    for j in h:

        i += 1

        

        img = cv2.imread(str(j), 0)

        test.append(cv2.resize(img, (59536, 1))[0])

        test.append(str(j))
test = np.array(test).reshape(-1, 2)

test = np.array(pd.DataFrame(test).sample(frac=1).reset_index(drop=True))
for k in test:

    if k[1].split('/')[6] == 'NORMAL':

        y_test.append(0)

    else:

        y_test.append(1)

    X_test.append(k[0])

    

X_test = pd.DataFrame(np.array(X_test).T)
i = 0



for h in [(val_path / 'NORMAL').glob('*.jpeg'), (val_path / 'PNEUMONIA').glob('*.jpeg')]:

    for j in h:

        i += 1

        

        img = cv2.imread(str(j), 0)

        val.append(cv2.resize(img, (59536, 1))[0])

        val.append(str(j))
val = np.array(val).reshape(-1, 2)

val = np.array(pd.DataFrame(val).sample(frac=1).reset_index(drop=True))
for k in val:

    if k[1].split('/')[6] == 'NORMAL':

        y_val.append(0)

    else:

        y_val.append(1)

    X_val.append(k[0])

    

X_val = pd.DataFrame(np.array(X_val).T)
X_train
X_test
X_val
X_train = X_train / 255.

X_test = X_test / 255.

X_val = X_val / 255.
X_train, y_train = np.array(X_train.T).reshape(5216, 244, 244, 1), np.array(y_train)

X_test, y_test = np.array(X_test.T).reshape(624, 244, 244, 1), np.array(y_test)
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
idg = ImageDataGenerator()

idg.fit(X_train)

train = idg.flow(X_train, y_train)



idg1 = ImageDataGenerator()

idg1.fit(X_test)

test = idg1.flow(X_test, y_test)
model = Sequential()

    

model.add(Conv2D(16, activation='relu', kernel_size=(2, 2), input_shape=(244, 244, 1)))

model.add(BatchNormalization())

model.add(Conv2D(16, activation='relu', kernel_size=(2, 2)))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

    

model.add(Conv2D(32, activation='relu', kernel_size=(2, 2)))

model.add(BatchNormalization())

model.add(Conv2D(32, activation='relu', kernel_size=(2, 2)))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

        

model.add(Conv2D(64, activation='relu', kernel_size=(2, 2)))

model.add(BatchNormalization())

model.add(Conv2D(64, activation='relu', kernel_size=(2, 2)))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

    

model.add(Conv2D(128, activation='relu', kernel_size=(2, 2)))

model.add(BatchNormalization())

model.add(Conv2D(128, activation='relu', kernel_size=(2, 2)))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

    

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(2, activation='sigmoid'))



    

checkpoint = ModelCheckpoint('vgg16_1.h5', monitor='val_acc')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=2, min_lr=0.001)

    

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=0.01))

history = model.fit_generator(generator=train, validation_data=test, callbacks=[checkpoint, reduce_lr], epochs=10)

    

for j in history.history:

    plt.plot(history.history[j])

    plt.title(str(j) + ' over epochs')

    plt.ylabel(j)

    plt.xlabel('epochs')

    plt.show()