# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset_train = pd.read_csv('../input/stockprice/Stock_Price_Train.csv')
dataset_train.head()
train = dataset_train.loc[: , ['Open']].values

train
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(0, 1))



train = scaler.fit_transform( train )



train.shape
x_train = []

y_train = []

timesteps = 50



for i in range( timesteps , 1258 ):

    x_train.append( train[i - timesteps : i , 0])

    y_train.append( train[ i , 0])    

    

x_train , y_train = np.array(x_train) , np.array(y_train)
x_train = np.reshape( x_train , (x_train.shape[0] , x_train.shape[1], 1))

x_train.shape
y_train
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import SimpleRNN

from keras.layers import Dropout
regressor = Sequential()



regressor.add( SimpleRNN( units = 50, activation = 'tanh', return_sequences=True , input_shape = ( x_train.shape[1], 1)))

regressor.add(Dropout(0.2))



regressor.add( SimpleRNN( units = 50, activation = 'tanh', return_sequences=True ))

regressor.add(Dropout(0.2))



regressor.add( SimpleRNN( units = 50, activation = 'tanh', return_sequences=True ))

regressor.add(Dropout(0.2))



regressor.add( SimpleRNN( units = 50 ))

regressor.add(Dropout(0.2))



regressor.add( Dense( units = 1))
regressor.compile( optimizer = 'Adam', loss = 'mean_squared_error')
hit = regressor.fit( x_train, y_train, epochs = 100 , batch_size = 32)
dataset_test = pd.read_csv('../input/stockprice/Stock_Price_Test.csv')
dataset_test.head()
real_stock_price = dataset_test.loc[: , ['Open']].values

real_stock_price
dataset_total = pd.concat( (dataset_train['Open'] , dataset_test['Open']) , axis = 0) 

inputs = dataset_total[ len(dataset_total) - len(dataset_test) - timesteps : ].values.reshape( -1 , 1)

inputs = scaler.transform( inputs )

inputs
x_test =  []



for i in range( timesteps , 70):

    x_test.append( inputs[i - timesteps :i , 0])

x_test = np.array(x_test)

x_test = np.reshape( x_test , (x_test.shape[0], x_test.shape[1], 1))
predicted_prices = regressor.predict( x_test )

predicted_prices = scaler.inverse_transform( predicted_prices )
plt.plot( real_stock_price , color = 'red', label = 'Real Google Stock Price' )

plt.plot( predicted_prices , color = 'blue', label = 'Predicted Google Stock Price' )

plt.title( 'Google Stock Price Prediction')

plt.xlabel( 'Time')

plt.ylabel( 'Google Stock Price')

plt.legend()

plt.show()
from keras.layers import LSTM

from keras.metrics import mean_squared_error
data = pd.read_csv('../input/stockprice/international-airline-passengers.csv', skipfooter = 5)
dataset = data.iloc[:,1].values

plt.plot(dataset)

plt.xlabel('Time')

plt.ylabel('Number of Passengers')

plt.title('International Airline Passenger')

plt.show()
dataset = dataset.reshape(-1,1)

dataset = dataset.astype('float32')

dataset.shape
scaler = MinMaxScaler( feature_range = (0,1))

dataset = scaler.fit_transform(dataset)
train_size = int( len(dataset) * 0.5 )

test_size = len(dataset) - train_size



train = dataset[0 :train_size, :]

test = dataset[train_size:, :]

print( 'Train Size: {} , Test Size: {}'.format( len(train) , len(test)))
timestamp = 10

dataX = []

dataY = []



for i in range(timestamp , len(train)):

    dataX.append( train[i-timestamp: i , 0])

    dataY.append( train[i , 0])

    

trainX = np.array( dataX )

trainY = np.array( dataY )
dataX = []

dataY = []



for i in range(timestamp , len(test)):

    dataX.append( test[i-timestamp: i , 0])

    dataY.append( test[i , 0])

    

testX = np.array( dataX )

testY = np.array( dataY )
trainX = np.reshape( trainX, ( trainX.shape[0] ,1 , trainX.shape[1] ))

testX  = np.reshape( testX,  ( testX.shape[0]  ,1 , testX.shape[1]  ))
model  = Sequential()

model.add( LSTM( units = 10 , input_shape = (1 , timestamp)))

model.add( Dense( 1 ))

model.compile( optimizer = 'adam', loss = 'mean_squared_error')

history = model.fit( trainX, trainY, epochs = 50, batch_size = 1 )
import math
train_predict = model.predict( trainX )

test_predict  = model.predict( testX )



train_predict = scaler.inverse_transform( train_predict )

trainY = scaler.inverse_transform( [trainY] )



test_predict = scaler.inverse_transform( test_predict )

testY = scaler.inverse_transform( [testY] )



trainScore = math.sqrt( mean_squared_error( trainY[0] , train_predict[:,0]))

print( 'Trainin Prediction Accuracy : ' , trainScore)



testScore = math.sqrt( mean_squared_error( testY[0] , test_predict[:,0]))

print( 'Trainin Prediction Accuracy : ' , testScore)
len(dataset)
trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:,:] = np.nan

trainPredictPlot[timestamp : len(train_predict) + timestamp, : ] = train_predict



testPredictPlot = np.empty_like(dataset)

testPredictPlot[:,:] = np.nan

testPredictPlot[len(train_predict) + (timestamp*2):len(dataset) , : ] = test_predict



plt.plot( scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()