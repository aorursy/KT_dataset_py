import numpy as np

import pandas as pd

from datetime import datetime

from pandas import Series

import matplotlib.pyplot as plt

import warnings

import math

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')



import os

print(os.listdir("../input"))

# load the dataset

dataframe = pd.read_excel('../input/otanchatram/madurai veg final merged (1).xlsx')

dataframe =dataframe.T

print(dataframe.shape)



#Naming the index

dataframe.index.name = 'Date'

print(dataframe.head())



#backing up orginal file

original = dataframe



#Index in DateTimeFormat

dataframe.index
#IndexDate is changed to column variable

dataframe.reset_index(level=0, inplace=True)
#Extract Features from Time

dataframe['Date'] = pd.to_datetime(dataframe.Date, format = '%Y-%m-%d')
#Checking missing values

print (dataframe.isnull().sum())

print (dataframe.columns)



#Forward filling the missing values.Assuming that the price of the product will be mostly nearer to the previous day price

dataframe.fillna(method='ffill',inplace=True)

print(dataframe.isnull().sum())



#Cleaning the values

#removing the empty column

dataframe.drop(dataframe.columns[-1],axis=1,inplace=True)

dataframe['Button Mushrooms / Button Kaalan'] = dataframe['Button Mushrooms / Button Kaalan'].map({'100-140':'100'})

print(dataframe.head())
# Data Preprocessing for TS

TS = original

TS.fillna(method='ffill',inplace=True)

print(TS.isnull().sum())



#removing the last empty column

TS.drop(TS.columns[-1],axis=1,inplace=True)

TS['Button Mushrooms / Button Kaalan'] = TS['Button Mushrooms / Button Kaalan'].map({'100-140':'100'})

print(TS.head())
#Tomato

df = dataframe[['Tomotto ottu / Thakkali Ottu']]

TS = TS[['Tomotto ottu / Thakkali Ottu']]
plt.figure(figsize = (16,8))

plt.plot(df)

plt.title("Time Series - Tomato price 2015")

plt.xlabel("Days")

plt.ylabel("Price")

plt.legend(loc = 'best')
dataframe['Day'] = dataframe['Date'].dt.day

dataframe['Month'] = dataframe['Date'].dt.month



dataframe.groupby('Day')['Tomotto ottu / Thakkali Ottu'].mean().plot.bar()
dataframe.groupby('Month')['Tomotto ottu / Thakkali Ottu'].mean().plot.bar()
temp = dataframe.groupby(['Month','Day'])['Tomotto ottu / Thakkali Ottu'].mean()

temp.plot(figsize =(15,5), title = "Tomato", fontsize = 14)
#Summary Statistics

X = TS.iloc[:,0].values

X1, X2 = X[0:104], X[104:]

mean1, mean2 = X1.mean(), X2.mean()

var1, var2 = X1.var(), X2.var()

print('mean1=%f, mean2=%f' % (mean1, mean2))

print('variance1=%f, variance2=%f' % (var1, var2))
#T-test

from scipy import stats



N = 104

a = X1

b = X2



## Calculate the Standard Deviation

#Calculate the variance to get the standard deviation

#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1

var_a = a.var(ddof=1)

var_b = b.var(ddof=1)



#std deviation

s = np.sqrt((var_a + var_b)/2)



# Calculate the t-statistics

t = (a.mean() - b.mean())/(s*np.sqrt(2/N))



## Compare with the critical t-value

#Degrees of freedom

df = 2*N - 2



#p-value after comparison with the t 

p = 1 - stats.t.cdf(t,df=df)



print("t = " + str(t))

print("p = " + str(2*p))



# Cross Checking with the internal scipy function

t2, p2 = stats.ttest_ind(a,b)

print("t = " + str(t2))

print("p = " + str(p2))
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=8,center=False).mean()

    rolstd = timeseries.rolling(window=8,center=False).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    print ('Results of Dickey-Fuller Test:')

    #adfuller() function accepts only 1d array of time series so first convert it using:

    dk = timeseries.iloc[:,0].values

    dftest = adfuller(dk, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
test_stationarity(TS)
#Moving average

moving_avg = TS.rolling(window=8,center=False).mean()

plt.plot(TS)

plt.plot(moving_avg, color='red')
#Note that since we are taking average of last 8 values, rolling mean is not defined for first 7 values

ts_moving_avg_diff = TS - moving_avg

ts_moving_avg_diff.dropna(inplace=True)

test_stationarity(ts_moving_avg_diff)
#Expotential weigghted moving average

expwighted_avg = TS.ewm(span=8,adjust=False).mean()

plt.plot(TS)

plt.plot(expwighted_avg, color='red')
ts_ewma_diff = TS - expwighted_avg

test_stationarity(ts_ewma_diff)
#Differencing

#we take the difference of the observation at a particular instant with that at the previous instant

ts_diff = TS - TS.shift()

plt.plot(ts_diff)
ts_diff.dropna(inplace=True)

test_stationarity(ts_diff)
#Multiplicative Decomposition



from pandas import Series

from matplotlib import pyplot

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(TS['Tomotto ottu / Thakkali Ottu'], model='multiplicative', freq=10)

trend = result.trend

seasonal = result.seasonal

residual = result.resid

result.plot()

pyplot.show()
ts_decompose = residual.to_frame()

ts_decompose = ts_decompose.dropna()
test_stationarity(ts_decompose)
#ARIMA

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

model = ARIMA(TS, order=(10, 1, 1))  

results_AR = model.fit(disp=-2)

plt.plot(ts_diff)

plt.plot(results_AR.fittedvalues, color='red')
results_AR = results_AR.fittedvalues.to_frame(name='Tomotto ottu / Thakkali Ottu')

test_stationarity(results_AR)
X = TS.values

size = int(len(X) * 0.66)

train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = list()

for t in range(len(test)):

    model = ARIMA(history, order=(10,1,0))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

# plot

pyplot.plot(test)

pyplot.plot(predictions, color='red')

pyplot.show()
# normalize the dataset

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

TS_log = scaler.fit_transform(TS)

# split into train and test sets

train_size = int(len(TS) * 0.67)

test_size = len(TS) - train_size

train, test = TS.iloc[0:train_size,:], TS.iloc[train_size:len(TS),:]

print(len(train), len(test))
import numpy

import matplotlib.pyplot as plt

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

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

dataset = TS

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

model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

# make predictions

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

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting

trainPredictPlot = numpy.empty_like(dataset)

trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = numpy.empty_like(dataset)

testPredictPlot[:, :] = numpy.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()