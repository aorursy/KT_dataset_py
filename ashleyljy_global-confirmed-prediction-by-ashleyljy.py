# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import requests

import matplotlib.pylab as plt

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')

data
sum_data1 = data.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

sum_data1[list(sum_data1.columns.values)].astype('int')

sum_data1.loc['sum'] = sum_data1.apply(lambda x: x.sum())



y = np.array(sum_data1.loc['sum'])



scaler = MinMaxScaler(feature_range=(0, 1))

dataf = scaler.fit_transform(y.reshape(-1, 1))

train = dataf.copy()





def create_dataset(dataset, timesteps=36, predict_size=1):  # creat the dataset

    datax = []  # create x

    datay = []  # create y

    for each in range(len(dataset) - timesteps - predict_steps):

        x = dataset[each:each + timesteps, 0]

        y = dataset[each + timesteps:each + timesteps + predict_steps, 0]

        datax.append(x)

        datay.append(y)

    return datax, datay





# create the train function and predict function

scaler = MinMaxScaler(feature_range=(0, 1))

dataf = scaler.fit_transform(dataf)

train = dataf.copy()

timesteps = 20  

predict_steps = 1  

length = 40  # predict steps

trainx, trainy = create_dataset(train, timesteps, predict_steps)

trainx = np.array(trainx)

trainy = np.array(trainy)



# transfer

trainx = np.reshape(trainx, (trainx.shape[0], timesteps, 1))  # transfer the shape to suitable for keras

# lstm training

model = Sequential()

model.add(LSTM(128, input_shape=(timesteps, 1), return_sequences=True))

model.add(Dropout(0.5))

model.add(LSTM(128, return_sequences=True))

# model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=False))

# model.add(Dropout(0.2))

model.add(Dense(predict_steps))

model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(trainx, trainy, epochs=50, batch_size=200)

# predict

# Because only 10 data can be predicted at a time, but 40 data must be predicted, the idea of circular prediction is adopted. The 10 data predicted each time are added to the data set to serve as the prediction x, and then the new 10 y are predicted, and then added to the prediction x list, and so on. Finally, 40 points are predicted.

predict_xlist = []  # create the dataset for prediction

predict_y = []  

predict_xlist.extend(dataf[dataf.shape[0] - timesteps:dataf.shape[0],

                     0].tolist())  # The last timesteps of the existing data are added to the list to predict the new value (such as the existing data from 1,2,3 to 80. Now we want to predict the following data, so add 80 data from 0 to 80 to the list, Predict the new value, that is, the data after 80 days)

while len(predict_y) < length:

    predictx = np.array(predict_xlist[

                        -timesteps:])  # Take timesteps data from the latest predict_xlist and predict new predict_steps data (because each predicted y will be added to the predict_xlist list, in order to predict the future value, so each time x is constructed to take the last timesteps in this list Data)

    predictx = np.reshape(predictx, (1, timesteps, 1))  # transfer the shape to be suitable for LSTM

    # predict the new value

    lstm_predict = model.predict(predictx)

    predict_xlist.extend(lstm_predict[0])

    # invert

    lstm_predict = scaler.inverse_transform(lstm_predict)

    predict_y.extend(lstm_predict[0])



y_predict = pd.DataFrame(predict_y, columns=["predict"])

# plot



y_true = np.array(sum_data1.loc['sum'])





min_d = min(y)

max_d = max(y)



y_predict['predict'] = y_predict['predict'].map(lambda x: x * (max_d - min_d) + min_d)



y_predict.index = range(81, 81 + 40)



plt.plot(y_true, c="g")

plt.plot(y_predict, c="r")

plt.show()

data2 = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv')

data2
sum_data2 = data2.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)

sum_data2[list(sum_data2.columns.values)].astype('int')

sum_data2.loc['sum'] = sum_data2.apply(lambda x: x.sum())



y = np.array(sum_data2.loc['sum'])



scaler = MinMaxScaler(feature_range=(0, 1))

dataf = scaler.fit_transform(y.reshape(-1, 1))

train = dataf.copy()





def create_dataset(dataset, timesteps=36, predict_size=1):  # creat the dataset

    datax = []  # create x

    datay = []  # create y

    for each in range(len(dataset) - timesteps - predict_steps):

        x = dataset[each:each + timesteps, 0]

        y = dataset[each + timesteps:each + timesteps + predict_steps, 0]

        datax.append(x)

        datay.append(y)

    return datax, datay





# create the train function and predict function

scaler = MinMaxScaler(feature_range=(0, 1))

dataf = scaler.fit_transform(dataf)

train = dataf.copy()

timesteps = 20  

predict_steps = 1  

length = 40  # predict steps

trainx, trainy = create_dataset(train, timesteps, predict_steps)

trainx = np.array(trainx)

trainy = np.array(trainy)



# transfer

trainx = np.reshape(trainx, (trainx.shape[0], timesteps, 1))  # transfer the shape to suitable for keras

# lstm training

model = Sequential()

model.add(LSTM(128, input_shape=(timesteps, 1), return_sequences=True))

model.add(Dropout(0.5))

model.add(LSTM(128, return_sequences=True))

# model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=False))

# model.add(Dropout(0.2))

model.add(Dense(predict_steps))

model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(trainx, trainy, epochs=50, batch_size=200)

# predict

# Because only 10 data can be predicted at a time, but 40 data must be predicted, the idea of circular prediction is adopted. The 10 data predicted each time are added to the data set to serve as the prediction x, and then the new 10 y are predicted, and then added to the prediction x list, and so on. Finally, 40 points are predicted.

predict_xlist = []  # create the dataset for prediction

predict_y = []  

predict_xlist.extend(dataf[dataf.shape[0] - timesteps:dataf.shape[0],

                     0].tolist())  # The last timesteps of the existing data are added to the list to predict the new value (such as the existing data from 1,2,3 to 80. Now we want to predict the following data, so add 80 data from 0 to 80 to the list, Predict the new value, that is, the data after 80 days)

while len(predict_y) < length:

    predictx = np.array(predict_xlist[

                        -timesteps:])  # Take timesteps data from the latest predict_xlist and predict new predict_steps data (because each predicted y will be added to the predict_xlist list, in order to predict the future value, so each time x is constructed to take the last timesteps in this list Data)

    predictx = np.reshape(predictx, (1, timesteps, 1))  # transfer the shape to be suitable for LSTM

    # predict the new value

    lstm_predict = model.predict(predictx)

    predict_xlist.extend(lstm_predict[0])

    # invert

    lstm_predict = scaler.inverse_transform(lstm_predict)

    predict_y.extend(lstm_predict[0])



y_predict = pd.DataFrame(predict_y, columns=["predict"])

# plot



y_true = np.array(sum_data2.loc['sum'])





min_d = min(y)

max_d = max(y)



y_predict['predict'] = y_predict['predict'].map(lambda x: x * (max_d - min_d) + min_d)



y_predict.index = range(81, 81 + 40)



plt.plot(y_true, c="g")

plt.plot(y_predict, c="r")

plt.show()