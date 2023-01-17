# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline 
import cv2
import os 
from tqdm import tqdm

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from random import shuffle



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data_dir = "../input/train"
test_data_dir = "../input/test"
def label(img):
    word_label = img.split('.')[-3]
    if word_label == "cat":
        return [1,0]
    elif word_label =="dog":
        return [0,1]
    

        
def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(train_data_dir)):
        lable = label(img)
        path = os.path.join(train_data_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        training_data.append([np.array(img), np.array(lable)])
    shuffle(training_data)
    return training_data

def create_test_data():
    test_data = []
    for img in tqdm(os.listdir(test_data_dir)):
        path = os.path.join(test_data_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        test_data.append([np.array(img)])
    return test_data
training_data = create_training_data()
testing_data = create_test_data()
training_data = training_data[:4000]
X_train = np.array([i[0] for i in training_data])
y_train = np.array([i[1] for i in training_data])

X_train= X_train/255.0
X_train = X_train.reshape(-1, 64,64,1)

X_test = np.array([i[0] for i in testing_data])

X_test = X_test/255.0
X_test = X_test.reshape(-1, 64,64,1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)
model = Sequential()
model.add(Conv2D(32, (5,5), padding = "same", activation = "relu", input_shape = (64,64,1)))
model.add(MaxPool2D(pool_size = (2,2), strides = (1,1)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (1,1)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dense(2, activation = "softmax"))

optimizer = Adam(lr = 0.0001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()
epochs = 50
batch_size = 200
model.fit(X_train, y_train, batch_size = batch_size,epochs = epochs, validation_data = (X_val, y_val), verbose=2)

model.predict(X_test, verbose =2)
