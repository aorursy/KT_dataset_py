import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
#
import plotly.graph_objs as go
import plotly.offline as offline
#
import warnings
warnings.filterwarnings('ignore')
#
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/GOOG_data.csv")
data.head()
data.shape
data.info()
trace_high = go.Scatter(x=data.date,
                        y=data.high,
                        
                        name = "Google High",
                        
                        line = dict(color = '#6699FF')
                       )
trace_low = go.Scatter( x=data.date,
                        y=data.low,
                        
                        name = "Google Low",
                        
                        line = dict(color = '#FF6633')
                       )
trace_open = go.Scatter( x=data.date,
                        y=data.open,
                        
                        name = "Google Open",
                        
                        line = dict(color = 'red')
                       )
trace_close = go.Scatter( x=data.date,
                        y=data.close,
                        
                        name = "Google Close",
                        
                        line = dict(color = 'black')
                       )
data_figure = [trace_open,trace_high, trace_low,trace_close]
layout = dict(
    
    title = 'Google Stock Price Data ',
    
    xaxis = dict(rangeselector = dict(buttons = list([dict(count = 1,
                                                           label = '1m',
                                                           step = 'month',
                                                           stepmode = 'todate',
                                                          visible = True),
                                                      
                                                  dict(count = 3,
                                                           label = '3m',
                                                           step = 'month',
                                                           stepmode = 'backward',
                                                          visible = True),
                                                      
                                                      dict(count = 6,
                                                           label = '6m',
                                                           step = 'month',
                                                           stepmode = 'backward',
                                                          visible = True),
                                                  
                                                      dict(step = 'all')])
                                     ),
                 
                 rangeslider=dict(visible = True),
                 type='date'
    )
)
fig = dict(data=data_figure, 
           layout=layout)

offline.iplot(fig)
data_temp = data.iloc[965:975,:]
trace = go.Candlestick(x = data_temp.date,                       
                       open = data_temp.open,                       
                       high = data_temp.high,                       
                       low = data_temp.low,                       
                       close = data_temp.close,
                      increasing = dict(fillcolor = 'greenyellow', 
                                         line = dict(color = 'green', 
                                                     width = 3
                                                    )),
                       decreasing = dict(fillcolor = 'lightcoral'),                       
                       whiskerwidth = 0.2)
data_figure_2 = [trace]
layout = dict(title = 'Google Stock Price Data ')
fig = dict(data=data_figure_2, 
           layout=layout)
offline.iplot(fig)
data_temp = data.iloc[875:975,:]
data_open = list(data_temp['open'])
dateList = list(data_temp['date'])
xList = []
yList = []
framesList = []
for i in range(len(dateList)):
    
    xList.append(dateList[i])
    yList.append(data_open[i])
    
    framesList.append(dict(data = [dict(x = xList.copy(), y = yList.copy())]))
#
playButton = dict(label = 'Play',
                  method= 'animate',
                  args= [None, 
                         dict(fromcurrent = True, 
                              transition = dict(duration = 200), 
                              frame = dict(duration = 100)
                             )
                        ]
                 )
#
pauseButton = dict(label = 'Pause',
                  method= 'animate',
                  args= [[None], dict(mode = 'immediate')]
                 )
#
layout = go.Layout(xaxis = dict(range = [dateList[0], dateList[-1]]), 
                   yaxis = dict(range = [0, 1 + max(data_open)]),
                   
                   updatemenus = [dict(type = 'buttons',
                                       buttons = [playButton, pauseButton]
                                       )
                                 ]
                  )
#
fig = dict(data=[{}], 
           layout=layout, 
           frames = framesList)

offline.iplot(fig)
# Split Data
dataset_train = data.loc[0:750,:]
dataset_test  = data.loc[750:,:]
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(dataset_train.loc[:,["open"]].values)
train_scaled
f,ax = plt.subplots(figsize = (30,7))
plt.plot(train_scaled)
plt.show()
X_train = []
y_train = []
timesteps = 50
for i in range(timesteps, 751):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X:",X_train)
print("X size:",X_train.size)
print("Y:",y_train)
print("Y size:",y_train.size)
# Initialising the RNN
regressor = Sequential()

# Adding the first RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 250, batch_size = 32)
dataset_test.head()
real_stock_price = dataset_test.loc[:,["open"]].values
real_stock_price
# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['open'], dataset_test['open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)  # min max scaler
inputs
X_test = []
for i in range(timesteps, 275):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualising the results
f,ax = plt.subplots(figsize = (30,7))
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
import numpy
import math
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
data = pd.read_csv("/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/GOOG_data.csv")
data.head()
# reshape
# Choice "open" feature:
dataset = data.iloc[:,1].values
dataset = dataset.reshape(-1,1) # (975,) sometimes can be problem
dataset = dataset.astype("float32")
dataset.shape
# scaling 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# train test split
train_size = int(len(dataset) * 0.75) # Split dataset 75% for train set, 25% for test set
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))
time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = numpy.array(dataX)
trainY = numpy.array(dataY)  
dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = numpy.array(dataX)
testY = numpy.array(dataY)  
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# model
model = Sequential()
model.add(LSTM(10, input_shape=(1, time_stemp))) # 10 lstm neuron(block)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore_vanilla = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore_vanilla))
# shifting train
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
f,ax = plt.subplots(figsize = (30,7))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
# Need to Preprocessing 
data = pd.read_csv("/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/GOOG_data.csv")
dataset = data.iloc[:,1].values
dataset = dataset.reshape(-1,1) # (975,) sometimes can be problem
dataset = dataset.astype("float32")
# scaling 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# train test split
train_size = int(len(dataset) * 0.75) # Split dataset 75% for train set, 25% for test set
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = numpy.array(dataX)
trainY = numpy.array(dataY)  
dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = numpy.array(dataX)
testY = numpy.array(dataY)  
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1,time_stemp)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(trainX, trainY, epochs=50, batch_size=1)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore_Stacked = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore_Stacked))
# shifting train
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
f,ax = plt.subplots(figsize = (30,7))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
# Need to Preprocessing 
data = pd.read_csv("/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/GOOG_data.csv")
dataset = data.iloc[:,1].values
dataset = dataset.reshape(-1,1) # (975,) sometimes can be problem
dataset = dataset.astype("float32")
# scaling 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# train test split
train_size = int(len(dataset) * 0.75) # Split dataset 75% for train set, 25% for test set
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = numpy.array(dataX)
trainY = numpy.array(dataY)  
dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = numpy.array(dataX)
testY = numpy.array(dataY)  
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(1,time_stemp)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(trainX, trainY, epochs=50, batch_size=1)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore_bidirectional = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore_bidirectional))
# shifting train
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
f,ax = plt.subplots(figsize = (30,7))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
# Need to Preprocessing 
data = pd.read_csv("/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/GOOG_data.csv")
dataset = data.iloc[:,1].values
dataset = dataset.reshape(-1,1) # (975,) sometimes can be problem
dataset = dataset.astype("float32")
# scaling 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# train test split
train_size = int(len(dataset) * 0.75) # Split dataset 75% for train set, 25% for test set
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = numpy.array(dataX)
trainY = numpy.array(dataY)  
dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = numpy.array(dataX)
testY = numpy.array(dataY)  
trainX = numpy.reshape(trainX, (trainX.shape[0], 1,1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1,1, testX.shape[1]))
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,1, time_stemp)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2, padding='same')))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(trainX, trainY, epochs=50,batch_size=1)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore_cnn = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore_cnn))
# shifting train
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
f,ax = plt.subplots(figsize = (30,7))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()