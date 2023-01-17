import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import signal
import cv2

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1) 
config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=config))
train_df = pd.read_csv('../input/mnist_train.csv')
test_df = pd.read_csv('../input/mnist_test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()
train_df['label'].value_counts().plot(kind='bar')
feat_cols = train_df.columns.tolist()[1:]
X_train = train_df[feat_cols].values.reshape(-1,28,28,1)/255
y_train = train_df['label'].values
X_test = test_df[feat_cols].values.reshape(-1,28,28,1)/255
y_test = test_df['label'].values
plt.imshow(X_test[234][:,:,0], cmap='Greys')
plt.gray()
plt.imshow([[1,-1]])
plt.show()
plt.imshow([[1],[-1]])
plt.show()
plt.imshow(X_train[12345][:,:,0])
plt.show()
plt.imshow(signal.convolve(X_test[234][:,:,0],[[1,-1]]))
plt.show()
plt.imshow(signal.convolve(X_test[234][:,:,0],[[1],[-1]]))
plt.show()
print(X_test.dtype, X_test.shape)
print(y_test.dtype, y_test.shape)
num_pixels = X_test.shape[1] * X_test.shape[2] # 28*28=784
X_train = X_train.reshape(-1, num_pixels).astype('float32')
X_test = X_test.reshape(-1, num_pixels).astype('float32')
print(X_train.dtype, X_train.shape)
print(X_test.dtype, X_test.shape)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.dtype)
print(y_train[0])
num_classes = len(set(train_df['label'].values)) # already one-hot encoded for labels, so cant use y_train
num_classes
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu')) # in=out hidde layer
model.add(Dense(num_classes, activation='softmax')) #output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train,y_train, validation_data=(X_test,y_test), batch_size=128, epochs=10, verbose=1)
# verbos=True = print training process
model.summary()
print(X_test.shape, type(X_test), X_test.dtype)
print(y_test.shape, type(y_test), y_test.dtype)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
print(X_test.shape, type(X_test), X_test.dtype)
print(y_test.shape, type(y_test), y_test.dtype)
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=3,input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train,y_train, validation_data=(X_test,y_test), batch_size=128, epochs=10, verbose=1)
model.summary()



