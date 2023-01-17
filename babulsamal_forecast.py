import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (12, 8)

plt.rcParams['font.size'] = 14

import datetime
df=pd.read_csv('../input/predice-el-futuro/train_csv.csv')

df

df.info()
train=df[0:56]

test=df[56:]

train,test
#Plotting data 

train.feature.plot(figsize=(15,8), title= 'All day features', fontsize=14) 

test.feature.plot(figsize=(15,8), title= 'All day features', fontsize=14) 

plt.show()

#Naive approach

dd= np.asarray(train.feature) 

y_hat = test.copy() 

y_hat['naive'] = dd[len(dd)-1] 

plt.figure(figsize=(12,8)) 

plt.plot(train.index, train['feature'], label='Train') 

plt.plot(test.index,test['feature'], label='Test') 

plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 

plt.legend(loc='best') 

plt.title("Naive Forecast") 

plt.show() 

 
from sklearn.metrics import mean_squared_error 

from math import sqrt 

rms = sqrt(mean_squared_error(test.feature, y_hat.naive)) 

print(rms) 

 

#simple average

y_hat_avg = test.copy() 

y_hat_avg['avg_forecast'] = train['feature'].mean() 

plt.figure(figsize=(12,8))

plt.plot(train['feature'], label='train') 

plt.plot(test['feature'], label='Test') 

plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')

plt.legend(loc='best') 

plt.show() 

 
# rms = sqrt(mean_squared_error(test.feature, y_hat_avg.avg_forecast)) 

print(rms) 
#moving average

y_hat_avg = test.copy() 

y_hat_avg['moving_avg_forecast'] = train['feature'].rolling(50).mean().iloc[-1]

plt.figure(figsize=(16,8))

plt.plot(train['feature'], label='Train') 

plt.plot(test['feature'], label='Test') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast') 

plt.legend(loc='best')

plt.show()

rms = sqrt(mean_squared_error(test.feature, y_hat_avg.moving_avg_forecast)) 

print("rms: %.4f" % rms)

#simple exponential smoothening

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

y_hat_avg = test.copy() 

fit2 = SimpleExpSmoothing(np.asarray(train['feature'])).fit(smoothing_level=0.6,optimized=False) 

y_hat_avg['SES'] = fit2.forecast(len(test))

plt.figure(figsize=(16,8)) 

plt.plot(train['feature'], label='train') 

plt.plot(test['feature'], label='test') 

plt.plot(y_hat_avg['SES'], label='SES') 

plt.legend(loc='best') 

plt.show() 

 
rms = sqrt(mean_squared_error(test.feature, y_hat_avg.SES)) 

print(rms) 
import statsmodels

print(statsmodels.__version__)
#Holt’s Linear Trend method

import statsmodels.api as sm 

sm.tsa.seasonal_decompose(train.feature, freq=3).plot() 

result = sm.tsa.stattools.adfuller(train.feature)

plt.show()

y_hat_avg = test.copy() 

 

fit1 = Holt(np.asarray(train['feature'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1) 

y_hat_avg['Holt_linear'] = fit1.forecast(len(test)) 

 

plt.figure(figsize=(16,8)) 

plt.plot(train['feature'], label='Train') 

plt.plot(test['feature'], label='Test') 

plt.plot(y_hat_avg['Holt_linear'], 

label='Holt_linear') 

plt.legend(loc='best') 

plt.show()

rms = sqrt(mean_squared_error(test.feature, y_hat_avg.Holt_linear)) 

print(rms)
y_hat_avg.Holt_linear.head()
test.feature.shape, y_hat_avg.Holt_linear.shape
predict=fit1.forecast(len(test))

predict
test['prediction']=predict
test.head()
#-Winters Method

y_hat_avg = test.copy() 

fit1 = ExponentialSmoothing(np.asarray(train['feature']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()

y_hat_avg['Holt_Winter'] = fit1.forecast(len(test)) 

plt.figure(figsize=(16,8)) 

plt.plot( train['feature'], label='Train') 

plt.plot(test['feature'], label='Test') 

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')

plt.legend(loc='best') 

plt.show()

rms = sqrt(mean_squared_error(test.feature, y_hat_avg.Holt_Winter))

print(rms) 

 

#check stationary

from statsmodels.tsa.stattools import adfuller 

def test_stationarity(timeseries):

        #Determing rolling statistics

    rolmean = timeseries.rolling(window=24).mean()

    rolstd = timeseries.rolling(window=24).std()

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

test_stationarity(train['feature'])
#removing trend

train_log = np.log(train['feature']) 

test_log = np.log(test['feature'])

moving_avg = train_log.rolling(24).mean()

plt.plot(train_log) 

plt.plot(moving_avg, color = 'red') 

plt.show()

train_log_moving_avg_diff = train_log - moving_avg
train_log_moving_avg_diff.dropna(inplace = True) 

test_stationarity(train_log_moving_avg_diff)
# differencing can help to make series stable and eliminate trend

train_log_diff = train_log - train_log.shift(1) 

test_stationarity(train_log_diff.dropna())
from statsmodels.tsa.seasonal import seasonal_decompose 

decomposition = seasonal_decompose(pd.DataFrame(train_log).feature.values, freq = 24) 



trend = decomposition.trend 

seasonal = decomposition.seasonal 

residual = decomposition.resid 



plt.subplot(411) 

plt.plot(train_log, label='Original') 

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

train_log_decompose['date'] = train_log.index 

train_log_decompose.set_index('date', inplace = True) 

train_log_decompose.dropna(inplace=True) 

test_stationarity(train_log_decompose[0])
from statsmodels.tsa.stattools import acf, pacf 

lag_acf = acf(train_log_diff.dropna(), nlags=25) 

lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')
#ACF and PACF plot

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

model = ARIMA(train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model 

results_AR = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(), label='original') 

plt.plot(results_AR.fittedvalues, color='red', label='predictions') 

plt.legend(loc='best') 

plt.show()
AR_predict=results_AR.predict(start=56, end=79) 

AR_predict=AR_predict.cumsum().shift().fillna(0) 

AR_predict1=pd.Series(np.ones(test.shape[0]) * np.log(test['feature'])[56], index = test.index) 

AR_predict1=AR_predict1.add(AR_predict,fill_value=0) 

AR_predict = np.exp(AR_predict1)

plt.plot(test['feature'], label = "test") 

plt.plot(AR_predict, color = 'red', label = "Predict") 

plt.legend(loc= 'best') 

plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, test['feature']))/test.shape[0])) 

plt.show()
model = ARIMA(train_log, order=(0, 1, 1))  # here the p value is zero since it is just the MA model 

results_MA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(), label='original') 

plt.plot(results_MA.fittedvalues, color='red', label='prediction') 

plt.legend(loc='best') 

plt.show()
MA_predict=results_MA.predict(start=56, end=79) 

MA_predict=MA_predict.cumsum().shift().fillna(0) 

MA_predict1=pd.Series(np.ones(test.shape[0]) * np.log(test['feature'])[56], index = test.index) 

MA_predict1=MA_predict1.add(MA_predict,fill_value=0) 

MA_predict = np.exp(MA_predict1)

plt.plot(test['feature'], label = "test") 

plt.plot(MA_predict, color = 'red', label = "Predict") 

plt.legend(loc= 'best') 

plt.title('RMSE: %.4f'% (np.sqrt(np.dot(MA_predict, test['feature']))/test.shape[0])) 

plt.show()
model = ARIMA(train_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(),  label='original') 

plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted') 

plt.legend(loc='best') 

plt.show()
def check_prediction_diff(predict_diff, given_set):

    predict_diff= predict_diff.cumsum().shift().fillna(0)

    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['feature'])[56], index = given_set.index)

    predict_log = predict_base.add(predict_diff,fill_value=0)

    predict = np.exp(predict_log)



    plt.plot(given_set['feature'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['feature']))/given_set.shape[0]))

    plt.show()

    

def check_prediction_log(predict_log, given_set):

    predict = np.exp(predict_log)

 

    plt.plot(given_set['feature'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['feature']))/given_set.shape[0]))

    plt.show()
ARIMA_predict_diff=results_ARIMA.predict(start=56, end=79)

check_prediction_diff(ARIMA_predict_diff, test)
import statsmodels.api as sm

y_hat_avg = test.copy() 

fit1 = sm.tsa.statespace.SARIMAX(train.feature, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit() 

y_hat_avg['SARIMA'] = fit1.predict(start=56, end=79, dynamic=True) 

plt.figure(figsize=(16,8)) 

plt.plot( train['feature'], label='feature') 

plt.plot(test['feature'], label='test') 

plt.plot(y_hat_avg['SARIMA'], label='SARIMA') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(test.feature, y_hat_avg.SARIMA)) 

print(rms)