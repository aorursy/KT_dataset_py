# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data(test_split = 0.3, seed = 2020)

## test_split parameter here says 30% of dataset should be kept for testing, so 70% will be used for training
## seed is used to ensure the random split is reproducible

## Examine the data
print(X_train.shape, X_test .shape)

## So we have 13 attributes or features as input for each sample.
print(y_train.shape, y_test.shape)

## Only one value has to be predicted as the target. 

## From the documentation using above link, we know that we are predicting median value of a house at a location in 1000s of $
print(np.amax(y_train), np.amin(y_train))
## We  can see that min value in training set is 5.0 and max value is 50.0 (in k$)

"""### Input normalization (aka Feature scaling):
- In this case, all features are continuous valued, so we can normalize them as shown here.
- But we shouldn't blindly normalize them in this way for any data. (Need to take care of categorical values and missing values)
"""

## Normalize inputs
X_train_mean = np.mean(X_train, axis = 0)
X_train_std = np.std(X_train, axis = 0)
X_train_normd = (X_train - X_train_mean)/(10**-8 + X_train_std)  ## adding a small number to denominator to avoid division by 0 errors

"""### Creating a model with Sequential class
- ```Fully connected``` layers are called ```Dense``` layers in keras
- We shall use ```relu``` activation
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential()  ## Initialize an empty sequential model
model.add(Input(shape = (13, )))   ## First add input layer, specify shape of 1 input sample
model.add(Dense(4, activation = 'relu'))  ## Add a hidden layer with 16 neurons, relu activation
model.add(Dense(2, activation = 'relu'))   ## Add a hidden layer with 8 neurons, relu activation
model.add(Dense(1, activation = 'relu'))   ## Add ouptut layer with 1 neuron (because we need to predict a single value), relu activation

## Look at summary of defined model
model.summary()

## Now compile the model - add optimizer, loss and metrics to track
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.01), loss = 'mean_squared_error', metrics = ['mse'])

## Now fit the model to training data - specify epochs, batch_size, validation_split
history = model.fit(X_train_normd, y_train, batch_size = 32, epochs = 100, validation_split = 0.1)

val_mse = history.history['val_mse']  ##The history attribute is a dictionary which stores metrics from each epoch
mse = history.history['mse']
import matplotlib.pyplot as plt
plt.plot(val_mse, color = 'red')  ## Plot the validation error
plt.plot(mse, color = 'blue') ## Plot training error
## You can see that the validation error saturate after 25 epochs or so. We don't need to train till 100 epochs (Early stopping)

## Now evaluate the model on test data
## But first normalize test data using same normalization that was used for training data

X_test_normd = (X_test - X_train_mean)/(10**-8 + X_train_std)
model.evaluate(X_test_normd, y_test)

