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
import numpy as np

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import math

from sklearn.metrics import mean_squared_error
data_import = pd.read_csv("/kaggle/input/google-stock-price/Google_Stock_Price_Train.csv")



dataset = data_import['Open']



scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(np.array(dataset).reshape(-1, 1))



TRAIN_SPLIT=0.8



training_size = int(len(dataset) * TRAIN_SPLIT)

test_size = len(dataset) - training_size

train_dataset, test_dataset = dataset[0:training_size, :], dataset[training_size:, :]
def create_dataset(data, time_step=1):

    dataX, dataY = [], []

    for i in range(len(data) - time_step - 1):

        a = data[i:(i + time_step), 0]  # i=0, 0,1,2,3-----99   100

        dataX.append(a)

        dataY.append(data[i + time_step, 0])

    return np.array(dataX), np.array(dataY)
time_step = 14

X_train, y_train = create_dataset(train_dataset, time_step)

X_test, y_test = create_dataset(test_dataset, time_step)



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = keras.models.Sequential([

    keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu',

                        input_shape=(time_step, 1)),

    #keras.layers.Dropout(0.5), # These were alternative layers I was testing out

    #keras.layers.LSTM(50),

    #keras.layers.Dropout(0.3),

    keras.layers.Flatten(),

    #keras.layers.GlobalMaxPooling

    keras.layers.Dense(1)

])





model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])



model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, verbose=1)
train_predict = model.predict(X_train)

test_predict = model.predict(X_test)



train_predict = scaler.inverse_transform(train_predict)

test_predict = scaler.inverse_transform(test_predict)



print(math.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predict)))



print(math.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict)))
plt.plot(scaler.inverse_transform(dataset))

plt.plot(range(time_step, len(train_predict) + time_step), train_predict, c='b')

plt.plot(range(len(train_predict) + 2*time_step, len(train_predict) + len(test_predict) + 2*time_step), test_predict, c='k')

plt.show()