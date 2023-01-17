# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data_history = [] #do not refresh this unless you want to delete the history pf parameters

# Any results you write to the current directory are saved as output.
epochs_custom = 2
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv("../input/1_MIN_ALL.txt", index_col=0, sep = ' ')
dataframe['Vol'] = dataframe['Vol'].str.replace("'", '')
dataframe = dataframe.tail(50000)
dataframe.head()
dataframe = dataframe[['Close']]
dataframe = dataframe.reset_index(drop = True)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=epochs_custom, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(1000*(trainY[0]), 1000*trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(1000*testY[0], 1000*testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predicZtions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
print(model.summary())
Predictions = []
Real = []
result = []
error= []

for i in range(0,len(testPredict)):
    if scaler.inverse_transform(testX[i])[0][look_back - 1] - testPredict[i] < 0:
        Predictions.append('Sell')
    else:
        Predictions.append('Buy')
    
    if scaler.inverse_transform(testX[i])[0][look_back - 1] - testY[0][i] < 0:
        Real.append('Sell')
    else:
        Real.append('Buy')
    
    if Predictions[i] == Real[i]:
        result.append(True)
    else:
        result.append(False)
    
    error.append(math.sqrt((scaler.inverse_transform(testX[i])[0][look_back - 1] - testPredict[i])*
            (scaler.inverse_transform(testX[i])[0][look_back - 1] - testPredict[i])))
freq = pd.DataFrame({'col':result})
freq.describe()
mse = pd.DataFrame({'col':error})
mse.describe()
accuracy = freq.describe()['col'][3]/freq.describe()['col'][0]
data_history.append([accuracy,epochs_custom,look_back,testScore,trainScore,mse.describe()['col'][3]])
df = pd.DataFrame(data_history,columns=['accuracy','epochs','look_back','trainScore','testScore','mse'])
df


