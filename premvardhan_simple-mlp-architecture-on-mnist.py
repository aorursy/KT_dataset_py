# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import utils
from keras.layers import Dropout
from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading train and test data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
# Shape of train and test data
print("Train data shape: {}\n Test data shape: {}".format(train_df.shape, test_df.shape))
train_df.head()
X_train = train_df.drop("label", axis = 1)
Y_train = train_df["label"]
X_test = test_df
print(X_train.shape, Y_train.shape, X_test.shape)
# If we observe each cell value is between 0-255 so we need to normalize it.
train_df[41970:41980] 
# Normalization 
# x = x - min(x) / max(x) - min(x)
X_train = X_train/255
X_test = X_test/255
# Here, we have class name(0- 9) for each image
print("Class label for 4th image is: ", Y_train[4])
# We need to convert it into 10 dimension vector
Y_train = utils.to_categorical(Y_train, 10)
print("After converting output into 10 dim is: ", Y_train[4])
# Define model parameter
input_dim = X_train.shape[1]
output_dim = 10
batch_size = 128
nb_epoch = 20
# 2 hidden layer with batch normalization and dropout layer
# MLP + relu + BN + Dropout 
model = Sequential()

model.add(Dense(434, activation = "relu", input_shape = (input_dim,), kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.039, seed = None)))
model.add(BatchNormalization())
model.add(Dropout(0,5))

model.add(Dense(391, activation = "relu", input_shape = (input_dim,), kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.039, seed = None)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(141, activation = "relu", input_shape = (input_dim,), kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.039, seed = None)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(output_dim, activation = "softmax"))

model.summary()
# Training
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model_fit = model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epoch, verbose = 1)
# Test Prediction
pred = model.predict_classes(X_test, verbose=0)
pred
# Train Score
score = model.evaluate(X_train, Y_train, verbose = 0) 
print(score)
print('Train score:', score[0]) 
print('Train accuracy:', score[1])
# Submission
sample_submission = pd.DataFrame({"ImageId": list(range(1, len(pred) + 1)), "Label" : pred})
sample_submission.to_csv("sample_submission.csv", index=False)