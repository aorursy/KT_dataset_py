import librosa

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import os

from PIL import Image

import pathlib

import csv

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

import keras

from keras import layers

from keras import layers

import keras

from keras.models import Sequential

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/birdcalldataset/dataset.csv')

data.head()

# Удаление ненужных столбцов

data = data.drop(['filename'],axis=1)

# Создание меток

genre_list = data.iloc[:, -1]

encoder = LabelEncoder()

y = encoder.fit_transform(genre_list)

# Масштабирование столбцов признаков

scaler = StandardScaler()

X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

# Разделение данных на обучающий и тестовый набор

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_class = len(data.label.unique())



model = Sequential()

model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(num_class, activation='softmax'))

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
classifier = model.fit(X_train,

                    y_train,

                    epochs=100,

                    batch_size=128)