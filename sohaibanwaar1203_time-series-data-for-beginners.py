# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Train.csv")

df.drop("ID",axis=1,inplace=True)

df.head()

print(df.describe(include="all"))

df.dtypes
df.Datetime = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 

df.dtypes
temp = df.iloc[1,0] # means at O coloum which is Date and and 1st Row 

temp.year

# By this we can easily seperate Date , Year , Time etc
# Now Seperating Date and time of Every Coloum in Our DataFrame

i = 0

year=[]

day =[]

month=[]

hour=[]

day_of_week=[]



while i < df.shape[0]:

    temp = df.iloc[i,0]

    year.append(temp.year)

    month.append(temp.month)

    day.append(temp.day)

    hour.append(temp.hour)

    day_of_week.append(temp.dayofweek)

    i +=1

train = df  

train["year"] = year

train["month"]= month

train["day of week"]= day_of_week

train["day"] = day

train["hour"]= hour

train.head()
import seaborn as sns

graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Year'] = train.year

sns.set_style("whitegrid")

ax = sns.barplot(x="Year", y="Count", data=graph).set_title("Caputuring the Trend")
# now we ae going to see our trends monthly with respect to year

graph = df[df["year"]==2014]

sns.set_style("whitegrid")

ax = sns.barplot(x="month", y="Count", data=graph).set_title("Caputuring the Trend")
# now we ae going to see our trends monthly with respect to year

graph = df[df["year"]==2013]

sns.set_style("whitegrid")

ax = sns.barplot(x="month", y="Count", data=graph).set_title("Caputuring the Trend")
# now we ae going to see our trends monthly with respect to year

graph = df[df["year"]==2012]

sns.set_style("whitegrid")

ax = sns.barplot(x="month", y="Count", data=graph).set_title("Caputuring the Trend")
graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Month'] = train.month



sns.set_style("whitegrid")

ax = sns.barplot(x="Month", y="Count", data=graph).set_title("Trends in Months")
graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Day'] = train.day



sns.set_style("whitegrid")

ax = sns.pointplot(x="Day", y="Count", data=graph).set_title("Trends in Days")
# We have Day too soo lets plot that too.

graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Day of Week'] = train["day of week"]

#fig, axs = plt.subplots(2,1)

sns.pointplot(x="Day of Week", y="Count", data=graph).set_title("Trends in WeekDays")
graph = pd.DataFrame()

graph['Count'] = train.Count

graph['Hour'] = train.hour



ax = sns.pointplot(x="Hour", y="Count", data=graph).set_title("Trends in Hour")
import matplotlib.pyplot as plt

train.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp



hourly = train.resample('H').mean()

daily = train.resample('D').mean()

weekly = train.resample('W').mean()

monthly = train.resample('M').mean()

fig, axs = plt.subplots(3,1)



sns.set(rc={'figure.figsize':(16.7,10.27)})

# sns.pointplot(data=hourly,y="Count",x=hourly.index,ax=axs[0]).set_title("Hourly")



sns.pointplot(data=daily,y="Count",x=daily.index,ax=axs[0]).set_title("Daily")

sns.pointplot(data=weekly,y="Count",x=weekly.index,ax=axs[1]).set_title("Weekly")

sns.pointplot(data=monthly,y="Count",x=monthly.index,ax=axs[2]).set_title("Monthly")

plt.show()




# Converting to daily mean

train = train.resample('D').mean()

train.to_csv("train_eda.csv")

train.iloc[0:11]
from pandas import Series

from statsmodels.tsa.stattools import adfuller

df=pd.read_csv("../input/Train.csv")

df.drop("ID",axis=1,inplace=True)

df.Datetime = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 





X=train.month



result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))
X=train.year



result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))
from statsmodels.tsa.stattools import adfuller 

def test_stationarity(timeseries):

        #Determing rolling statistics

    #rolmean = pd.rolling_mean(timeseries, window=24) # 24 hours on each day

    rolmean = timeseries.rolling(24).mean()

    rolstd = timeseries.rolling(24).std()

    #rolmean = timeseries.rolling(24).mean()

        #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

        #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])



    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)

from matplotlib.pylab import rcParams 

rcParams['figure.figsize'] = 20,10

test_stationarity(df['Count'])
Train_log = np.log(train['Count']) 





moving_avg = Train_log.rolling(24).mean()

plt.plot(Train_log) 

plt.plot(moving_avg, color = 'red') 

plt.show()
train_log_moving_avg_diff = Train_log - moving_avg

train_log_moving_avg_diff.dropna(inplace = True) 

test_stationarity(train_log_moving_avg_diff)

train_log_diff = Train_log - Train_log.shift(1) 

test_stationarity(train_log_diff.dropna())
from statsmodels.tsa.seasonal import seasonal_decompose 

decomposition = seasonal_decompose(pd.DataFrame(Train_log).Count.values, freq = 24) 



trend = decomposition.trend 

seasonal = decomposition.seasonal 

residual = decomposition.resid 



plt.subplot(411) 

plt.plot(Train_log, label='Original') 

plt.legend(loc='best') 

plt.subplot(412) 

plt.plot(trend, label='Trend') 

plt.legend(loc='best') 

plt.subplot(413) 

plt.plot(seasonal,label='Seasonality') 

plt.legend(loc='best') 

plt.subplot(414) 

plt.plot(residual, label='Residuals') 

plt.legend(loc='best') 

plt.tight_layout() 

plt.show()
train_log_decompose = pd.DataFrame(residual) 

train_log_decompose['date'] = Train_log.index 

train_log_decompose.set_index('date', inplace = True)

train_log_decompose.dropna(inplace=True) 

test_stationarity(train_log_decompose[0])
from statsmodels.tsa.stattools import acf, pacf 

lag_acf = acf(train_log_diff.dropna(), nlags=25) 

lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')
plt.plot(lag_acf) 

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 

plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 

plt.title('Autocorrelation Function') 

plt.show() 

plt.plot(lag_pacf) 

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 

plt.title('Partial Autocorrelation Function') 

plt.show()

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(Train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model 

results_AR = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(), label='original') 

plt.plot(results_AR.fittedvalues, color='red', label='predictions') 

plt.legend(loc='best') 

plt.show()
AR_predict=results_AR.predict(start="2014-06-25", end="2014-09-25")

AR_predict=AR_predict.cumsum().shift().fillna(0) 

AR_predict1=pd.Series(np.ones(train.shape[0]) * np.log(train['Count'])[0], index = train.index) 

AR_predict1=AR_predict1.add(AR_predict,fill_value=0) 

AR_predict = np.exp(AR_predict1)

plt.plot(train['Count'], label = "Valid") 

plt.plot(AR_predict, color = 'red', label = "Predict") 

plt.legend(loc= 'best') 

plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, train['Count']))/train.shape[0]))

plt.show()
model = ARIMA(Train_log, order=(0, 1, 2))  # here the p value is zero since it is just the MA model 

results_MA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(), label='original') 

plt.plot(results_MA.fittedvalues, color='red', label='prediction') 

plt.legend(loc='best') 

plt.show()
# We cannot seperate time series randomly because we have to maintain the sequence to time stamps. Becuase we have to predict

# fureter time counts. 

# split into train and test sets

import matplotlib.pyplot as plt

import pandas

import math

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

model = Sequential()

dataframe = pd.read_csv('../input/Train.csv', usecols=[2], engine='python', skipfooter=3)

dataset = dataframe.values

dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(len(train), len(test))

import numpy

def create_dataset(dataset, look_back=1):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back), 0]

		dataX.append(a)

		dataY.append(dataset[i + look_back, 0])

	return numpy.asarray(dataX), numpy.asarray(dataY)


from sklearn.preprocessing import MinMaxScaler

look_back=1

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))





model.add(LSTM(4, input_shape=(1, look_back)))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=1)
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