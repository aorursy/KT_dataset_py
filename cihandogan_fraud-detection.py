import numpy as np 

import pandas as pd

import os

from sklearn.model_selection import train_test_split

import random

import tensorflow as tf

from tensorflow.keras.layers import Dense ,Dropout

from tensorflow.keras.models import Sequential

import cv2

import matplotlib.pyplot as plt
path = "/kaggle/input/creditcardfraud/creditcard.csv"

data = pd.read_csv(path)

ds = data.values

random.shuffle(ds)

X,y = (ds[:,:30],ds[:,30:31])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10)

])



model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size= 100, epochs=10, validation_data=(X_test,y_test))