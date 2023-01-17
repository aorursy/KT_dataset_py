# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y


digit_file = "../input/train.csv"
digit_data = np.loadtxt(digit_file, skiprows=1, delimiter=',')
x, y = prep_data(digit_data, train_size=50000, val_size=5000)
import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

def build_network(num_actions):
    
    inputs = Input(shape=(img_rows, img_cols, 1,))
    model = Convolution2D(filters=12, kernel_size=(3,3),activation='relu')(inputs)
    
    model = Convolution2D(filters=12, kernel_size=(3,3), activation='relu',)(model)
    model = Flatten()(model)
    model = Dense(activation='relu', units=10)(model)
    q_values = Dense(units=num_actions, activation='linear')(model)
    m = Model(input=inputs, output=q_values)
    return  m

d_model = build_network(num_classes)
# Your code to compile the model in this cell
d_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
# Your code to fit the model here
d_model.fit(x,y, batch_size=100, epochs=4, validation_split=0.2)
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

with open('../input/test.csv', 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    # This skips the first row of the CSV file.
    # csvreader.next() also works in Python 2.
    next(csvreader)
    for data in csvreader:
        # The first column is the label
        #label = data[0]
        # The rest of columns are pixels
        pixels = data
        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels, dtype='int64')
        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))
        # Plot
        img = plt.imshow(pixels, cmap='gray')
        plt.show()
        break # This stops the loop, I just want to see one


digit_test_file = "../input/test.csv"
digit_test_data = np.loadtxt(digit_test_file, skiprows=1, delimiter=',')

def prep_test_data(raw):
    x = raw
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x
test = prep_test_data(digit_test_data)

#Need to debug and implement prediction part
predict = d_model.predict(test)
print(predict.shape)