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
# import pytse_client as tse

# tickers = tse.download(symbols="فملی", write_to_csv=True)
data = pd.read_csv('../input/fameli-daily/fameli.csv')

data.drop(columns = ['date','count','close','value'],inplace=True)

data_np = data.to_numpy()

data
n1 = int(data_np.shape[0] * 0.9)

n2 = int((data_np.shape[0] - n1)/2)

x_train = data_np[:n1]

x_val = data_np[n1: n1 + n2]

x_test = data_np[n1 + n2:]
from sklearn import preprocessing



minmax_scale = preprocessing.MinMaxScaler().fit(x_train)

x_train_n = minmax_scale.transform(x_train)

x_val_n = minmax_scale.transform(x_val)

x_test_n = minmax_scale.transform(x_test)

def slicing_50(x, history_points):

    

    sliced_data = np.array([x[i  : i + history_points] for i in range(len(x) - history_points)])

    

    labels = np.array([x[:,3][i + history_points] for i in range(len(x) - history_points)])

    

    return sliced_data, labels

history_points = 50

x_train_n, y_train = slicing_50(x_train_n, history_points)

x_val_n, y_val = slicing_50(x_val_n, history_points)

x_test_n, y_test = slicing_50(x_test_n, history_points)
import keras

import tensorflow as tf

from keras.models import Model

from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate

from keras import optimizers

np.random.seed(4)

import tensorflow

tensorflow.random.set_seed(4)



lstm_input = Input(shape=(history_points, 5), name='lstm_input')

x = LSTM(50, name='lstm_0')(lstm_input)

x = Dropout(0.2, name='lstm_dropout_0')(x)

x = Dense(64, name='dense_0')(x)

x = Activation('sigmoid', name='sigmoid_0')(x)

x = Dense(1, name='dense_1')(x)

output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)



adam = optimizers.Adam(lr=0.0005)



model.compile(optimizer=adam, loss='mse')
from keras.utils import plot_model

plot_model(model)
from keras.callbacks import ModelCheckpoint

mcp_save = ModelCheckpoint('./stocks_price.h5', save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(x=x_train_n, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_data=(x_val_n, y_val), callbacks=[mcp_save])
model.load_weights('./stocks_price.h5')

evaluation = model.evaluate(x_test_n, y_test)

print(evaluation)
y_train_real = np.array([x_train[:,3][i + history_points] for i in range(len(x_train) - history_points)])

scale_back = preprocessing.MinMaxScaler().fit(np.expand_dims(y_train_real, -1))
y_test_predicted = model.predict(x_test_n)

y_test_predicted = scale_back.inverse_transform(y_test_predicted)

y_test_real = np.array([x_test[:,3][i + history_points] for i in range(len(x_test) - history_points)])

real_mse = np.square(np.mean(y_test_real - y_test_predicted))

print(real_mse)
import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)



start = 0

end = -1



real = plt.plot(y_test_real[start:end], label='real')

pred = plt.plot(y_test_predicted[start:end], label='predicted')



plt.legend(['Real', 'Predicted'])



plt.show()