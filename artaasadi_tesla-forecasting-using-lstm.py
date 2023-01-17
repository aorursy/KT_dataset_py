import pandas as pd

import numpy as np

import datetime

import time



from sklearn.preprocessing import StandardScaler



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Dropout

from tensorflow.keras.losses import MSE

from sklearn.metrics import mean_squared_error as mse



import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots
data = pd.read_csv('../input/tesla-stock-data-from-2010-to-2020/TSLA.csv')
data.head()
data = data[['Date', 'Close']]

data.head()
scaler = StandardScaler()

close = np.array(data['Close'])

close = scaler.fit_transform(close.reshape(-1, 1))
close
#We are going to define our test and train dataset using create dataset which makes new x and y datasets

def create_dataset(dataset, look_back=1):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back), 0]

		dataX.append(a)

		dataY.append(dataset[i + look_back, 0])

	return np.array(dataX), np.array(dataY)
train_size = int(len(close) * 0.7) 

test_size = len(close) - train_size

train, test = close[0:train_size, :], close[train_size:len(close), :]
look_back = 10

trainX, trainY = create_dataset(train, look_back)  

testX, testY = create_dataset(test, look_back)



# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#Simple LSTM

model = Sequential()

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=100, validation_split=0.33 , batch_size=1, verbose=1)
predicted = model.predict(testX)

print('MSE for predicted', mse(np.array(testY).reshape(-1),

                               np.array(predicted).reshape(-1)))
# make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])
trainPredictPlot = np.empty_like(close)

trainPredictPlot[:, :] = np.nan

trainPredictPlot = trainPredictPlot.reshape(-1)

trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.reshape(-1)

# shift test predictions for plotting

testPredictPlot = np.empty_like(close)

testPredictPlot[:, :] = np.nan

testPredictPlot = testPredictPlot.reshape(-1)

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(close)-1] = testPredict.reshape(-1)
fig = go.Figure()

fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Tesla original",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=data['Date'], y=trainPredictPlot, name="Predicted train",

                         line_color='dimgray'))



fig.add_trace(go.Scatter(x=data['Date'], y=testPredictPlot, name="Predicted test",

                         line_color='darkviolet'))



fig.update_layout(title_text='Tesla stock price trough time',

                  xaxis_rangeslider_visible=True)

py.offline.iplot(fig)
look_back = 10

trainX, trainY = create_dataset(train, look_back)  

testX, testY = create_dataset(test, look_back)



# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



#Simple LSTM

model = Sequential()

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=100, validation_split=0.33 , batch_size=1, verbose=1)



predicted = model.predict(testX)

print('MSE for predicted', mse(np.array(testY).reshape(-1),

                               np.array(predicted).reshape(-1)))



# make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])





trainPredictPlot = np.empty_like(close)

trainPredictPlot[:, :] = np.nan

trainPredictPlot = trainPredictPlot.reshape(-1)

trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.reshape(-1)

# shift test predictions for plotting

testPredictPlot = np.empty_like(close)

testPredictPlot[:, :] = np.nan

testPredictPlot = testPredictPlot.reshape(-1)

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(close)-1] = testPredict.reshape(-1)







fig = go.Figure()

fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Tesla original",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=data['Date'], y=trainPredictPlot, name="Predicted train",

                         line_color='dimgray'))



fig.add_trace(go.Scatter(x=data['Date'], y=testPredictPlot, name="Predicted test",

                         line_color='darkviolet'))



fig.update_layout(title_text='Tesla stock price trough time',

                  xaxis_rangeslider_visible=True)

py.offline.iplot(fig)
look_back = 10

trainX, trainY = create_dataset(train, look_back)  

testX, testY = create_dataset(test, look_back)



# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



#Simple LSTM

model = Sequential()

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adagrad')

model.fit(trainX, trainY, epochs=100, validation_split=0.33 , batch_size=1, verbose=1)



predicted = model.predict(testX)

print('MSE for predicted', mse(np.array(testY).reshape(-1),

                               np.array(predicted).reshape(-1)))



# make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])





trainPredictPlot = np.empty_like(close)

trainPredictPlot[:, :] = np.nan

trainPredictPlot = trainPredictPlot.reshape(-1)

trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.reshape(-1)

# shift test predictions for plotting

testPredictPlot = np.empty_like(close)

testPredictPlot[:, :] = np.nan

testPredictPlot = testPredictPlot.reshape(-1)

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(close)-1] = testPredict.reshape(-1)







fig = go.Figure()

fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Tesla original",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=data['Date'], y=trainPredictPlot, name="Predicted train",

                         line_color='dimgray'))



fig.add_trace(go.Scatter(x=data['Date'], y=testPredictPlot, name="Predicted test",

                         line_color='darkviolet'))



fig.update_layout(title_text='Tesla stock price trough time',

                  xaxis_rangeslider_visible=True)

py.offline.iplot(fig)