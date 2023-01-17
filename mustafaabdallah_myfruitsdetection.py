import os

import cv2

import numpy as np

import pandas as pd



from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt

%matplotlib inline
train_path='../input/fruits/fruits-360_dataset/fruits-360/Training'

test_path='../input/fruits/fruits-360_dataset/fruits-360/Test'



train_labels=os.listdir(train_path)

test_labels=os.listdir(test_path)
train_generator=ImageDataGenerator(rescale=1./255)

test_generator=ImageDataGenerator(rescale=1./255)



tarin_data = train_generator.flow_from_directory(train_path, batch_size=32, classes=train_labels, target_size=(64,64))

test_data = test_generator.flow_from_directory(test_path, batch_size=32, classes=train_labels, target_size=(64,64))
cnn = Sequential()



cnn.add(Conv2D(16, kernel_size=(3, 3), input_shape = (64, 64, 3), padding = "same", activation = "relu"))

cnn.add(MaxPooling2D())



cnn.add(Conv2D(32, kernel_size=(5,5), padding='same', activation='relu'))

cnn.add(MaxPooling2D())



cnn.add(Conv2D(64, kernel_size=(5,5), padding='same', activation='relu'))

cnn.add(MaxPooling2D())



cnn.add(Flatten())



cnn.add(Dropout(0.25))

cnn.add(Dense(256, activation = "relu"))

cnn.add(Dense(len(train_labels), activation = "softmax"))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_cnn = cnn.fit(tarin_data ,epochs=2, verbose=1, validation_data=test_data)
plt.plot(history_cnn.history['accuracy'])

plt.plot(history_cnn.history['val_accuracy'])
score = cnn.evaluate(test_data)

score