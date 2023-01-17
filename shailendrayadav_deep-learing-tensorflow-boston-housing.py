import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from tensorflow import keras

data = np.load('../input/boston_housing.npz')
#(X_train,y_train),(X_test,y_test)=boston_housing.load_data() #for inbuilt datasets

from sklearn.model_selection import train_test_split

X_train, y_train,X_test,y_test = train_test_split(data['x'], data['y'], test_size=0.2, random_state=42)



X_train = tf.keras.utils.normalize(X_train)

X_test = tf.keras.utils.normalize(X_test)
X_train.shape
X_test.shape
n_cols =X_train.shape[1]  # number of columns
# The input shape specifies the number of rows and columns in the input. The number of columns in our input is stored in ‘n_cols’. 

#There is nothing after the comma which indicates that there can be any amount of rows.
#model -sequential

from keras.models import Sequential

model=Sequential()

#adding layers for the model

from keras.layers import Dense

model.add(Dense(1000,activation="relu",input_shape=(n_cols,)))

model.add(Dense(1000,activation="relu"))
model.summary()
#training

import keras.metrics 

model.compile(optimizer="adam",loss="mean_squared_error") # mse- mean sqaure error
model.fit(X_train,y_train,validation_split=0.2, epochs=30)