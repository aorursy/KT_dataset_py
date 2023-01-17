# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
print("Shape of train.csv file ",train_df.shape)
test_df = pd.read_csv("../input/test.csv")
print("Shape of test.csv file", test_df.shape)
train_df.sample(5)
test_df.sample(5)
Y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values
X_test = test_df.iloc[:, 0:].values
X_train = X_train.astype('float32')
print("training data: {}".format(X_train.shape))
print("training labels {}".format(Y_train.shape))
img_width=28
img_height=28
img_depth=1

X_train = X_train.reshape(len(X_train), img_width, img_height, img_depth)
X_test = X_test.reshape(len(X_test), img_width, img_height, img_depth)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)
plt.figure(figsize=(12,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Class {}".format(Y_train[i]))
Y_train = to_categorical(Y_train, num_classes=10)
model = Sequential()
model.add(Convolution2D(32,(3, 3), activation='relu', input_shape=(28,28, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
saved_path='cnn_model.h5'
model_checkpoint = ModelCheckpoint(filepath=saved_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=6, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.001, mode='auto')
callback_list = [model_checkpoint, early_stop, reduce_lr]

train_history = model.fit(X_train, Y_train, batch_size=32, 
                          epochs=15, verbose=1, callbacks=callback_list, validation_split=0.2)
model = load_model('cnn_model.h5')
labels = model.predict_classes(X_test)
plt.figure(figsize=(12,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("predicted class {}".format(labels[i]))
submit_df = pd.DataFrame({'ImageId': list(range(1, 28001)), 'Label': labels})
submit_df.to_csv('submit.csv', index=False)