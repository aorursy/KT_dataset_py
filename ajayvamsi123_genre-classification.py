import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import os
import keras
import h5py
import librosa
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import drive
drive.mount('/content/gdrive')
X = np.load('/kaggle/input/data1/x_gtzan_npy.npy')
y = np.load('/kaggle/input/data1/y_gtzan_npy.npy')
y.shape
X.shape
X = X[0:7400]
y = y[0:7400]
# One hot encoding of the labels
y = to_categorical(y)
X = X[:10000]
y = y[:10000]
y.shape
X_stack = np.squeeze(np.stack((X,) * 3, -1))
X_stack.shape
X_train, X_test, y_train, y_test = train_test_split(X_stack, y, test_size=0.3, random_state=42, stratify = y)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Histogram for train and test 
values, count = np.unique(np.argmax(y_train, axis=1), return_counts=True)
plt.bar(values, count)

values, count = np.unique(np.argmax(y_test, axis=1), return_counts=True)
plt.bar(values, count)
plt.show()
input_shape = X_train[0].shape
num_genres = 4

input_tensor = Input(shape=input_shape)
vgg16 = VGG16(include_top=False, weights='imagenet',input_shape = input_shape)
top = Sequential()
top.add(Flatten(input_shape=vgg16.output_shape[1:]))
top.add(Dense(256, activation='relu'))
top.add(Dropout(0.5))
top.add(Dense(num_genres, activation='softmax'))
print(vgg16.output_shape[1:])
print(vgg16.input)
print(vgg16.output)
print(top(vgg16.output))
model = Model(inputs=vgg16.input, outputs=top(vgg16.output))

print(input_tensor)
top = Sequential()
top.add(Flatten(input_shape=vgg16.output_shape[1:]))
top.add(Dense(256, activation='relu'))
top.add(Dropout(0.5))
top.add(Dense(num_genres, activation='softmax'))
print(vgg16.output_shape[1:])
print(vgg16.input)
print(vgg16.output)
print(top(vgg16.output))
model = Model(inputs=vgg16.input, outputs=top(vgg16.output))
print(top(vgg16.output))
print(top)
model.summary()
for layer in model.layers :
   print(layer,"----->",layer.trainable)
len(model.layers)
for i, layer in enumerate(model.layers):
        
        if(i <= 5):
            layer.trainable = False
        else:
            layer.trainable = True
for layer in model.layers :
   print(layer,"----->",layer.trainable)
early_stopping_callback = EarlyStopping(monitor='val_acc', patience=4)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
hist = model.fit(X_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_data=(X_test, y_test),
          shuffle=True,
          callbacks=[early_stopping_callback])
for layers in top.layers:
   print(layers)
