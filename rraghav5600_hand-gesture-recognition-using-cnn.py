import tensorflow as tf

import os

import cv2

import random

import numpy as np

import time

import matplotlib.pyplot as plt



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout

from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
PATH = '/kaggle/input'

IMG_SIZE = 200

training_data = []
for dirname, _, filenames in os.walk(PATH):

    for filename in filenames:

        img_path = os.path.join(dirname, filename)

        img = cv2.imread(img_path, 0)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        training_data.append([img, img_path.split('_')[2]])

        

random.shuffle(training_data)
X = []

y = []



for features,label in training_data:

    X.append(features)

    if ord(label)<60: y.append(ord(label)-ord('0'))

    else: y.append(ord(label)-ord('a')+10)



X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X = X/255.0

y = np.array(y)
model = Sequential()



model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))



model.add(Flatten())



model.add(Dense(128))

model.add(Dropout(rate=0.5))



model.add(Dense(128))

model.add(Dropout(rate=0.5))



model.add(Dense(36))

model.add(Activation('softmax'))



model.compile(loss='sparse_categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
NAME = "Hand-Gesture-Recognition-CNN"

tensorboard = TensorBoard(log_dir=NAME)
trainer = model.fit(X, y, epochs=10, validation_split=0.2,

          callbacks = [tensorboard])
epoch = [i+1 for i in range(len(trainer.history['accuracy']))]
plt.plot(epoch, trainer.history['accuracy'],)

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.show()
plt.plot(epoch,trainer.history['loss'])

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
plt.plot(epoch, trainer.history['val_accuracy'])

plt.xlabel('epoch')

plt.ylabel('val_accuracy')

plt.show()
plt.plot(epoch, trainer.history['val_loss'])

plt.xlabel('epoch')

plt.ylabel('val_loss')

plt.show()