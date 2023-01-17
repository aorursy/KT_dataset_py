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
import tensorflow as tf
import csv
from keras.utils import to_categorical

train_file = '../input/train.csv'
test_file = '../input/test.csv'
submission_file = '../input/sample_submission.csv'
output_file = 'submission.csv'

train = pd.read_csv(train_file, encoding ='utf-8')
train_y = to_categorical(train['label'])
train_x = ((train.drop('label', axis = 1)).values) / 255.0

test = pd.read_csv(test_file, encoding ='utf-8')
test_x = test / 255.0

submission = pd.read_csv(submission_file, encoding ='utf-8')

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(train_x, train_y, validation_split=0.3, epochs =10)

# Calculate predictions: predictions
predictions = model.predict(test_x)

# columns of submission is ImageId, Label

for i in range(0,len(submission)):
    digit = predictions[i].argmax(axis=0)
    submission.loc[i,'Label'] = digit
    
submission.to_csv(output_file, index = False)
print(os.listdir("./"))