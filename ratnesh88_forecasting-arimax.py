import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dateparse = lambda d : dt.datetime.strptime(d, '%Y-%m')

data = pd.read_csv("../input/AirPassengers.csv", index_col='Month',date_parser=dateparse)

data.info()
data.head()
data.describe()
#x = [dt.datetime.strptime(d,'%Y-%m') for d in data.index]

#data.set_index = [ str(d.year)+"-"+str(d.month)+"-"+str(d.day) for d in x]

data.index
data['#Passengers']['1949-01-01']
plt.figure(figsize=(12,8))

plt.plot(data['#Passengers'])
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    plt.figure(figsize=(12,8))

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=12).mean()

    rolstd = timeseries.rolling(window=12).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
test_stationarity(data['#Passengers'])
ts_log = np.log(data['#Passengers'])

plt.figure(figsize=(12,8))

plt.plot(ts_log)
test_stationarity(np.log(data['#Passengers']))
moving_avg = np.log(data['#Passengers']).rolling(12).mean()

plt.plot(np.log(data['#Passengers']))

plt.plot(moving_avg, color='red')
ts_log_moving_avg_diff = np.log(data['#Passengers']) - moving_avg

ts_log_moving_avg_diff.head(12)
ts_log_moving_avg_diff.dropna(inplace=True)

test_stationarity(ts_log_moving_avg_diff)
expwighted_avg = np.log(data['#Passengers']).ewm(halflife=12).mean()

plt.plot(np.log(data['#Passengers']))

plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = np.log(data['#Passengers']) - expwighted_avg

test_stationarity(ts_log_ewma_diff)
moving_avg = ts_log.rolling(12).mean()

plt.plot(ts_log)

plt.plot(moving_avg, color='red')
ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diff.head(12)
expwighted_avg = ts_log.ewm(halflife=12).mean()

plt.plot(ts_log)

plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = ts_log - expwighted_avg

test_stationarity(ts_log_ewma_diff)
ts_log_diff = ts_log - ts_log.shift()

plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)

test_stationarity(ts_log_diff)
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

# fit model

model = ARIMA(ts_log, order=(5,1,0))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# plot residual errors

residuals = pd.DataFrame(model_fit.resid)

residuals.plot()

plt.show()
residuals.plot(kind='kde')

plt.show()

print(residuals.describe())
validation_size =int(len(ts_log)*0.66)

X = data['#Passengers'].astype('float')

train, test = X[0:validation_size], X[validation_size:len(X)]

history = [x for x in train]

predictions = list()
for t in range(len(test)):

	model = ARIMA(history, order=(5,1,0))

	model_fit = model.fit(disp=0)

	output = model_fit.forecast()

	yhat = output[0]

	predictions.append(yhat)

	obs,y = test[t],test.index[t]

	history.append(obs)

	print('predicted=%f, expected=%f, month=%s' % (yhat, obs, y))
error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)
plt.figure(figsize=(12,8))



print(type(predictions),type(history))

plt.plot(predictions, color='red',label="Predictions")

plt.plot(history,label="Label")