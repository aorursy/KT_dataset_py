import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten
import cv2
from random import shuffle
import matplotlib.pyplot as plt

import os

img_path = '../input/train'
IMG_SIZE = 100
def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'cat':
        return [1,0]
    if word_label == 'dog':
        return [0,1]
    
def create_train_data():
    training_data = []
    for img in os.listdir(img_path):
        label = label_img(img)
        path = os.path.join(img_path,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
        shuffle(training_data)
    return training_data
train = create_train_data()
X = np.array([i[0] for i in train])
X = X.reshape(-1,100,100,1)
X = X/255
Y = np.array([i[1] for i in train])
model = Sequential()
model.add(Conv2D(64, kernel_size = (3,3), input_shape=(100,100,1), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Dropout(0.30))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Dropout(0.30))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.30))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer='adam',
              loss= 'binary_crossentropy',
              metrics=['accuracy', 'mae'])
Fit = model.fit(X, Y, epochs = 5, batch_size = 32, validation_split = 0.30)
plt.plot(Fit.history['loss'])
plt.plot(Fit.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()
plt.plot(Fit.history['acc'])
plt.plot(Fit.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()