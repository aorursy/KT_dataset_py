from math import sqrt

from numpy import concatenate

import pandas as pd

from datetime import datetime



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers import LSTM



from matplotlib import pyplot

import plotly.offline as py

import plotly.graph_objs as go

import numpy as np

import seaborn as sns



py.init_notebook_mode(connected=True)

%matplotlib inline
# https://coinmarketcap.com/

data = pd.read_csv(filepath_or_buffer="../input/data_bitcoin/data_bitcoin.csv", 

                   index_col="Date")
data.head()
data.tail()
btc_trace = go.Scatter(x=data.index, y=data['Close'], name= 'Price')

py.iplot([btc_trace])
data['Close'].replace(0, np.nan, inplace=True)

data['Close'].fillna(method='ffill', inplace=True)
btc_trace = go.Scatter(x=data.index, y=data['Close'], name= 'Price')

py.iplot([btc_trace])
#last_week = data[-8:-1]

#data      = data[0:-8]
#btc_trace = go.Scatter(x=last_week.index, y=last_week['Close'], name= 'Price')

#py.iplot([btc_trace])
values = data['Close'].values.reshape(-1,1)

values = values.astype('float32')

scaler = StandardScaler()

scaled = scaler.fit_transform(values)
train_size = len(scaled)-30#int(len(scaled) * 0.9)

test_size = len(scaled) - train_size

train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]

print(len(train), len(test))
last_week
def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset) - look_back):

        a = dataset[i:(i + look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    print(len(dataY))

    return np.array(dataX), np.array(dataY)
train
look_back = 2

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



testX.shape
model = Sequential()

model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))

#model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

model.summary()
history = model.fit(trainX, trainY, 

                    epochs=300, batch_size=100, 

                    validation_data=(testX, testY), 

                    shuffle=False)
pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()
yhat = model.predict(testX)

pyplot.plot(yhat, label='predict')

pyplot.plot(testY, label='true')

pyplot.legend()

pyplot.show()
yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))

testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))

print('Test RMSE: %.3f' % rmse)
pyplot.plot(yhat_inverse, label='predict')

pyplot.plot(testY_inverse, label='actual', alpha=0.5)

pyplot.legend()

pyplot.show()
predictDates = data.tail(len(testX)).index
testY_reshape = testY_inverse.reshape(len(testY_inverse))

yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))
actual_chart = go.Scatter(x=predictDates, y=testY_reshape, name= 'Actual Price')

predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict Price')

py.iplot([predict_chart, actual_chart])
scaled = scaler.transform(last_week)

last_weekX = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))

testY_reshape = testY_inverse.reshape(len(testY_inverse))

yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))
yhat = model.predict(last_weekX)

pyplot.plot(yhat, label='predict')

pyplot.plot(last_week, label='true')

pyplot.legend()

pyplot.show()