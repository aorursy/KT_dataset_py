# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, BatchNormalization, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Load the data
train_data_file = pd.read_csv("../input/train.csv")
test_data_file = pd.read_csv("../input/test.csv")
# Split the data away from the labels
X_train = train_data_file.drop(['label'], axis=1)
Y_train = train_data_file['label']
# Reshape data
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_train = X_train/255.0
test_data_file = test_data_file.values.reshape(-1, 28, 28, 1)
test_data_file = test_data_file/255.0

# One hot encode the labels
Y_train = keras.utils.to_categorical(Y_train, 10)
# Create a validation set of data to cross validate
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

print("Number of training samples ", X_train.shape[0])
print("Number of validation samples ", X_val.shape[0])
model = Sequential()

# Convolutional Layer #1
model.add(Conv2D(filters=32, kernel_size=(1,1), kernel_initializer='truncated_normal', padding='same', activation='relu', input_shape=(28,28,1)))

# Convolutional Layer #2
model.add(Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='truncated_normal', padding='same', activation='relu'))
model.add(BatchNormalization())

# Convolutional Layer #3
model.add(Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='truncated_normal', padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

# Convolutional Layer #4
model.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='truncated_normal', padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

# Convolutional Layer #5
model.add(Conv2D(filters=128, kernel_size=(3,3), kernel_initializer='truncated_normal', padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

# Flatten
model.add(Flatten())

# Fully Connected #1
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Fully Connected #2
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Softmax Output
model.add(Dense(10, activation='softmax'))
# Image Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1)

datagen.fit(X_train)
model.summary()
opt = keras.optimizers.Adam(1e-4)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), validation_data=(X_val, Y_val),
          steps_per_epoch=len(X_train) / 32,
          callbacks=[early_stop, checkpointer],
          epochs = 100, verbose=1)
# Predict
model = keras.models.load_model("weights.hdf5")

results = model.predict(test_data_file)
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
