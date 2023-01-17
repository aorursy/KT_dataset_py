# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
data = "../input/flowers/flowers"

folders = os.listdir("../input/flowers/flowers")
print(folders)
image_names = []
train_labels = []
train_images = []

size = 64,64

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            raw_img = cv2.imread(os.path.join(data,folder,file))
            img = cv2.resize(raw_img,size)
            train_images.append(img)
        else:
            continue
type(train_images)
train = np.array(train_images)
train.shape
%matplotlib inline
print('Label: ', train_labels[2000])
train.shape

train = train.astype('float32')/255.0
train.shape
label_dummies = pd.get_dummies(train_labels)

labels = label_dummies.values.argmax(1)
len(labels)
unique_list = list(zip(train,labels))
random.shuffle(unique_list)
train,labels = zip(*unique_list)

train = np.array(train)
labels = np.array(labels)
x_train , x_test , y_train , y_test = train_test_split(train , labels , test_size=0.3)

# Plot the Neural network fitting history
def history_plot(fit_history, n):
    plt.figure(figsize=(18, 12))
    
    plt.subplot(211)
    plt.plot(fit_history.history['loss'][n:], color='slategray', label = 'train')
    plt.plot(fit_history.history['val_loss'][n:], color='#4876ff', label = 'valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Loss Function');  
    
    plt.subplot(212)
    plt.plot(fit_history.history['acc'][n:], color='slategray', label = 'train')
    plt.plot(fit_history.history['val_acc'][n:], color='#4876ff', label = 'valid')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")    
    plt.legend()
    plt.title('Accuracy');
model = Sequential()
#model.add(Flatten(input_shape=(64,64,3)))
#model.add(Dense(128, activation='tanh'))
model.add(Conv2D(32,(2,2), padding='same', input_shape=(64,64,3)))
#model.add(Conv2D(64,(2,2)))
model.add(Activation('relu'))


model.add(Conv2D(128,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(164,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size=100, epochs=50, validation_data=(x_test, y_test))
history_plot(history, 0)
