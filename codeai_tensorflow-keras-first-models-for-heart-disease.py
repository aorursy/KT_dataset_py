import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

import os

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
path = '../input'

heart_file = 'heart.csv'

file_path = os.listdir(path)

data_heart = pd.read_csv(os.path.join('../input',heart_file))
data_heart.shape
data_heart.columns
for i in data_heart.index:

    if (data_heart.loc[i].isnull().sum() != 0):

        print('Missing value at ', i)

print('Done!')
data_heart_features = data_heart.loc[:,data_heart.columns!='target']

data_heart_target = data_heart.iloc[:,-1]
X_train_all,X_test_all,y_train_all,y_test_all = train_test_split(data_heart_features,data_heart_target,test_size=0.20,random_state=42)
X_train_all.shape
X_test_all.shape
model = keras.Sequential([

    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train_all.keys())]),

    keras.layers.Dense(64, activation=tf.nn.relu),

    keras.layers.Dense(1)

])
optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',

                optimizer=optimizer,

                metrics=['accuracy'])
model.summary()
model.fit(X_train_all,y_train_all,epochs=1000)
print(model.evaluate(X_test_all,y_test_all))
data_features = data_heart.loc[:,['cp','slope','exang','thal']]

data_target = data_heart.iloc[:,-1]
X_train_four,X_test_four,y_train_four,y_test_four = train_test_split(data_features,data_target,test_size=0.20,random_state=42)
model = keras.Sequential([

    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train_four.keys())]),

    keras.layers.Dense(64, activation=tf.nn.relu),

    keras.layers.Dense(1)

])
model.compile(loss='mean_squared_error',

                optimizer='adam',

                metrics=['accuracy'])
model.fit(X_train_four,y_train_four,epochs=1000)
print(model.evaluate(X_test_four,y_test_four))