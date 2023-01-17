# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dense, Flatten, Lambda  
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
# Load raw data
def prep_data(raw):
    X = (raw.iloc[:,1:].values).astype('float32')
    y = raw.iloc[:, 0].values.astype('int32')
    
    X = X.reshape(X.shape[0], 28, 28)
    
    return X, y

raw_train = pd.read_csv('../input/train.csv')
X_train, y_train = prep_data(raw_train)
# Visualing data
def visualize(X, y):
    for i in range(6, 9):
        plt.subplot(330 + (i+1))
        plt.imshow(X[i], cmap=plt.get_cmap('gray'))
        plt.title(y[i])
        
visualize(X_train, y_train)
# Reshape to (?, rows, cols, channels) -> /255
# Convert y to one hot encoding labels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# print(X_train[0])
# X_train /= 255.0
# print(X_train[0])
y_train = to_categorical(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

def standardize(x):
    mean = np.mean(x).astype(np.float32)
    std = np.std(x).astype(np.float32)
    return (x - mean) / deviation

num_class = y_train.shape[1]

gen = ImageDataGenerator(width_shift_range=0.1,
                        height_shift_range=0.1,
                        rotation_range=10,
                        shear_range=0.3)

train_gen = gen.flow(X_train, y_train, batch_size=50)
val_gen = gen.flow(X_val, y_val, batch_size=50)
# Building model
model = Sequential()
# model.add(Lambda(standardize, input_shape=(28, 28, 1)))

model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization(axis=1))

model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(BatchNormalization(axis=1))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization(axis=1))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2))

model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=num_class, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Training
model.fit_generator(train_gen,
                   steps_per_epoch=756,
                   validation_data=val_gen,
                   validation_steps=84,
                   epochs=3)
# Submit predictions to Kaggle
raw_test = pd.read_csv('../input/test.csv')
X_test = raw_test.values.astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("mnist_submit_4.csv", index=False, header=True)