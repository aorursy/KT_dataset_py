import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import re

from sklearn.model_selection import train_test_split

def convert_boardtomatrix(board):

    matrix = np.zeros(shape=[board.shape[0], board.shape[1]])

    for i in range(board.shape[0]):

        for j in range(board.shape[1]):

            # a represents the value of 8 neighbours

            a = []

            if i - 1 >= 0:

                a.append(1 if board[i - 1][j] == 1 else 0)

            if j - 1 >= 0:

                a.append(1 if board[i][j - 1] == 1 else 0)

            if i - 1 >= 0 and j - 1 >= 0:

                a.append(1 if board[i - 1][j - 1] == 1 else 0)

            if i + 1 < board.shape[0]:

                a.append(1 if board[i + 1][j] == 1 else 0)

            if j + 1 < board.shape[1]:

                a.append(1 if board[i][j + 1] == 1 else 0)

            if i + 1 < board.shape[0] and j + 1 < board.shape[1]:

                a.append(1 if board[i + 1][j + 1] == 1 else 0)

            if i - 1 >= 0 and j + 1 < board.shape[1]:

                a.append(1 if board[i - 1][j + 1] == 1 else 0)

            if i + 1 < board.shape[0] and j - 1 >= 0:

                a.append(1 if board[i + 1][j - 1] == 1 else 0)



            matrix[i][j] = np.sum(a)

    return matrix









def filter_string(str_arr, regex): 

    p = re.compile(regex)

    return [ s for s in str_arr if p.match(s) ]
df = pd.read_csv('../input/conways-reverse-game-of-life-2020/train.csv')
start_cols = filter_string(df.columns, 'start')

stop_cols = filter_string(df.columns, 'stop')
X = df[start_cols][:10000].to_numpy()

Y = df[stop_cols][:10000].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

x_train = np.array([convert_boardtomatrix(tm.reshape([25, 25])) for tm in x_train]).reshape([-1, 25, 25, 1])

x_test = np.array([convert_boardtomatrix(tm.reshape([25, 25])) for tm in x_test]).reshape([-1, 25, 25, 1])
from tensorflow.keras import models, layers, losses

def build_CNN():

    model = models.Sequential()

    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(25, 25, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (2, 2), activation='relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(625, activation='sigmoid'))

    model.compile(optimizer='adam',

              loss='MSE',

              metrics=['accuracy'])

    return model
model = build_CNN()

model.summary()
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
plt.plot(history.history['accuracy'])