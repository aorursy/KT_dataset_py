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
import matplotlib.pyplot as plt

spy = pd.read_csv('/kaggle/input/SPY.csv', usecols=['Close'])

spy['Close_SMA_30'] = spy.iloc[:,0].rolling(window=30).mean()

plt.plot(spy['Close_SMA_30'])
from sklearn.preprocessing import MinMaxScaler

# Normalization

scaler = MinMaxScaler(feature_range=(0, 1))

ds = scaler.fit_transform(spy['Close_SMA_30'].values.reshape(-1,1))

# ds = spy['Close_SMA_30'].values.reshape(-1,1)

train_edge, valid_edge = int(ds.shape[0]*0.7), int(ds.shape[0]*0.7)+int(ds.shape[0]*0.15)

train, valid, test = ds[29:train_edge, :], ds[train_edge:valid_edge, :], ds[valid_edge:, :]

print(ds.shape, train.shape, valid.shape, test.shape)

def create_xy(ds):

    x, y = [], []

    for i in range(len(ds)-1):

        x.append(ds[i: i+1, 0])

        y.append(ds[i+1, 0])

    return np.array(x), np.array(y)



train_x, train_y = create_xy(train)

valid_x, valid_y = create_xy(valid)

test_x, test_y = create_xy(test)



train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))

valid_x = np.reshape(valid_x, (valid_x.shape[0], 1, valid_x.shape[1]))

test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

print(train_x.shape, valid_x.shape, test_x.shape, train_y.shape, valid_y.shape, test_y.shape)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(LSTM(4, input_shape=(1, 1)))

model.add(Dense(1))

model.summary()



model.compile(loss='mae', optimizer=Adam(lr=0.0005))

print(train_x.shape, valid_x.shape, test_x.shape, train_y.shape, valid_y.shape, test_y.shape)

model.fit(train_x, train_y, epochs=18, verbose=1, validation_data=(valid_x, valid_y))
predict = model.predict(test_x)

predict = scaler.inverse_transform(predict)

test_y = scaler.inverse_transform(np.reshape(test_y, (-1,1)))

print(test_y.shape, test_y)

print(predict.shape, predict)

plt.plot(test_y)

plt.plot(predict)

plt.show()