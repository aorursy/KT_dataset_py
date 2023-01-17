# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from math import sqrt

from datetime import datetime

from pandas import Series

import seaborn as sns

%matplotlib inline

import warnings

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20,10

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the data sets



df_train = pd.read_csv(r"../input/into-the-future/train.csv")

df_test = pd.read_csv(r'../input/into-the-future/test.csv')

# Copy of original data

train_original = df_train.copy()

test_original = df_test.copy()
df_test.head()
df_train.head()
df_train.shape
# Converting to datetime format

df_train['Datetime'] = pd.to_datetime(df_train.time,format='%Y-%m-%d %H:%M:%S') 

df_test['Datetime'] = pd.to_datetime(df_test.time,format='%Y-%m-%d %H:%M:%S') 

test_original['Datetime'] = pd.to_datetime(test_original.time,format='%Y-%m-%d %H:%M:%S')
# Start and end date for train

df_train['Datetime'].min(),df_train['Datetime'].max()
# Start and end date for test

df_test['Datetime'].min(),df_test['Datetime'].max()
# Creating new features

for i in (df_train, df_test, test_original):

    i['year'] = i.Datetime.dt.year 

    i['month'] = i.Datetime.dt.month 

    i['day'] = i.Datetime.dt.day

    i['Hour'] = i.Datetime.dt.hour

    i['minute'] = i.Datetime.dt.minute

    i['seconds'] = i.Datetime.dt.second
# Droping id,time column from train

df_train = df_train.drop(['id', 'time'],axis = 1)
df_train.Timestamp = pd.to_datetime(df_train.Datetime,format='%Y-%m-%d %H:%M:%S') 

df_train.index =df_train.Timestamp



df_test.Timestamp = pd.to_datetime(df_test.Datetime,format='%Y-%m-%d %H:%M:%S') 

df_test.index = df_test.Timestamp 
# Hourly time series

hourly = df_train.resample('H').mean()



# Minutly time series

Minutly = df_train.resample('min').mean()



# Secondly time series

secondly = df_train.resample('S').mean()
fig, axs = plt.subplots(2,1)



hourly.feature_2.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0])

Minutly.feature_2.plot(figsize=(15,8), title= 'Minutly', fontsize=14, ax=axs[1])



plt.show()
df_train.tail()
# Training and Validation data split



Train=df_train.loc['2019-03-19 00:00:00':'2019-03-19 01:00:00']

valid=df_train.loc['2019-03-19 01:00:00':'2019-03-19 01:33:00']
Train.feature_2.plot(figsize=(15,8), title= 'Minutes TSA', fontsize=14, label='train') 

valid.feature_2.plot(figsize=(15,8), title= 'Minutes TSA', fontsize=14, label='valid') 

plt.xlabel("Datetime") 

plt.ylabel("Feature_2") 

plt.legend(loc='best') 

plt.show()
ts = df_train['feature_2']

plt.figure(figsize=(16,8)) 



plt.plot(ts,label='Feature_2')

# plt.plot(df['feature_1'],label='Other_feture')

plt.title('Time Series') 

plt.xlabel("Time") 

plt.ylabel("number_count") 

plt.legend()
import seaborn as sns

sns.jointplot(x='feature_1',y='feature_2',data=df_train)

df_train.groupby('year')['feature_2'].mean().plot.bar()
df_train.groupby('Hour')['feature_2'].mean().plot.bar()
df_train.groupby(['month','minute'])['feature_2'].mean().plot.bar()
# Moving Average 

y_hat_avg = valid.copy()

y_hat_avg['moving_avg_forecast'] = Train['feature_2'].rolling(10).mean().iloc[-1] # average of last 10 observations.

plt.figure(figsize=(15,5)) 

plt.plot(Train['feature_2'], label='Train')

plt.plot(valid['feature_2'], label='Valid')

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations')

plt.legend(loc='best')

plt.show()



y_hat_avg = valid.copy()

y_hat_avg['moving_avg_forecast'] = Train['feature_2'].rolling(20).mean().iloc[-1] # average of last 20 observations.

plt.figure(figsize=(15,5))

plt.plot(Train['feature_2'], label='Train')

plt.plot(valid['feature_2'], label='Valid')

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations')

plt.legend(loc='best')

plt.show()



y_hat_avg = valid.copy()

y_hat_avg['moving_avg_forecast'] = Train['feature_2'].rolling(50).mean().iloc[-1] # average of last 50 observations.

plt.figure(figsize=(15,5))

plt.plot(Train['feature_2'], label='Train')

plt.plot(valid['feature_2'], label='Valid')

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations')

plt.legend(loc='best')

plt.show()
# Calculating rmse value

rmse = sqrt(mean_squared_error(valid.feature_2, y_hat_avg.moving_avg_forecast))

print('RMSE value is :',rmse)
# Simple exponential smoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

y_hat_avg = valid.copy()

fit2 = SimpleExpSmoothing(np.asarray(Train['feature_2'])).fit(smoothing_level=0.6,optimized=False)

y_hat_avg['SES'] = fit2.forecast(len(valid))

plt.figure(figsize=(16,8))

plt.plot(Train['feature_2'], label='Train')

plt.plot(valid['feature_2'], label='Valid')

plt.plot(y_hat_avg['SES'], label='SES')

plt.legend(loc='best')

plt.show()
# calculating rmse value

rmse = sqrt(mean_squared_error(valid.feature_2, y_hat_avg.SES))

print('RMSE value is :',rmse)
# Holt's Linear trend model

y_hat_avg = valid.copy()



fit1 = Holt(np.asarray(Train['feature_2'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)

y_hat_avg['Holt_linear'] = fit1.forecast(len(valid))



plt.figure(figsize=(16,8))

plt.plot(Train['feature_2'], label='Train')

plt.plot(valid['feature_2'], label='Valid')

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')

plt.legend(loc='best')

plt.show()
# calculating rmse value

rmse = sqrt(mean_squared_error(valid.feature_2, y_hat_avg.Holt_linear))

print('RMSE value is :',rmse)
# Function for checking stationarity of data (Dickey Fuller test for stationarity)

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = pd.Series.rolling(timeseries, 24).mean()

    rolstd = pd.Series.rolling(timeseries, 24).std()

    

    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='green',label='Original')

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
test_stationarity(train_original['feature_2'])


#Estimating & Eliminating Trend



Train_log = np.log(Train['feature_2'])

valid_log = np.log(valid['feature_2'])
# Taking rolling mean

moving_avg = pd.Series.rolling(Train_log, 20).mean()

plt.plot(Train_log)

plt.plot(moving_avg, color = 'red')

plt.show()
# Removing trend

train_log_moving_avg_diff = Train_log - moving_avg



train_log_moving_avg_diff.dropna(inplace = True)

test_stationarity(train_log_moving_avg_diff)
train_log_diff = Train_log - Train_log.shift(1)

test_stationarity(train_log_diff.dropna())

# Decomposing the time series into trend, seasonality and residual

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(pd.DataFrame(Train_log).feature_2.values, freq = 24)



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
#ACF & PACF plots

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(train_log_diff.dropna(), nlags=20)

lag_pacf = pacf(train_log_diff.dropna(), nlags=20, method='ols')

rcParams['figure.figsize']  =  10, 5

#Plot ACF:

plt.subplot(121)

plt.plot(lag_acf)

plt.axhline(y = 0, linestyle = '--', color = 'gray')

plt.axhline(y = -1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color = 'gray')

plt.axhline(y = 1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color = 'gray')

plt.xticks(np.arange(0,22,2))

plt.title('Autocorrelation Function')            



#Plot PACF

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y = 0, linestyle = '--', color = 'gray')

plt.axhline(y = -1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color = 'gray')

plt.axhline(y = 1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color = 'gray')

plt.xticks(np.arange(0,22,2))

plt.title('Partial Autocorrelation Function')

            

plt.tight_layout()            
from statsmodels.tsa.arima_model import ARIMA



# AR model

model = ARIMA(Train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model

results_AR = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(), label='original')

plt.plot(results_AR.fittedvalues, color='blue', label='predictions')

plt.legend(loc='best')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-train_log_diff.dropna())**2))

plt.show()
AR_predict = results_AR.predict(start="2019-03-19 01:00:00", end="2019-03-19 01:33:00")

AR_predict = AR_predict.cumsum().shift().fillna(0)

AR_predict1 = pd.Series(np.ones(valid.shape[0]) * np.log(valid['feature_2'])[0], index = valid.index)

AR_predict1 = AR_predict1.add(AR_predict,fill_value=0)

AR_predict = np.exp(AR_predict1)
plt.plot(valid['feature_2'], label = "Valid")

plt.plot(AR_predict, color = 'red', label = "Predict")

plt.legend(loc= 'best')

plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, valid['feature_2']))/valid.shape[0]))

plt.show()
# MA model

ARIMA = ARIMA(Train_log, order=(0, 1, 2))  # here the p value is zero since it is just the MA model

results_MA = ARIMA.fit(disp=-1)  

plt.plot(train_log_diff.dropna(), label='original')

plt.plot(results_MA.fittedvalues, color='red', label='prediction')

plt.legend(loc='best')

plt.show()
MA_predict=results_MA.predict(start="2019-03-19 01:00:00", end="2019-03-19 01:33:00")

MA_predict=MA_predict.cumsum().shift().fillna(0)

MA_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['feature_2'])[0], index = valid.index)

MA_predict1=MA_predict1.add(MA_predict,fill_value=0)

MA_predict = np.exp(MA_predict1)
plt.plot(valid['feature_2'], label = "Valid")

plt.plot(MA_predict, color = 'red', label = "Predict")

plt.legend(loc= 'best')

plt.title('RMSE: %.4f'% (np.sqrt(np.dot(MA_predict, valid['feature_2']))/valid.shape[0]))

plt.show()
from statsmodels.tsa.arima_model import ARIMA



# ARIMA model

model = ARIMA(Train_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(),  label='original')

plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted')

plt.legend(loc='best')

plt.show()
# Rescaling the values to original

def check_prediction_diff(predict_diff, given_set):

    predict_diff= predict_diff.cumsum().shift().fillna(0)

    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['feature_2'])[0], index = given_set.index)

    predict_log = predict_base.add(predict_diff,fill_value=0)

    predict = np.exp(predict_log)

    

    plt.plot(given_set['feature_2'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['feature_2']))/given_set.shape[0]))

    plt.show()
def check_prediction_log(predict_log, given_set):

    predict = np.exp(predict_log)

    

    plt.plot(given_set['feature_2'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['feature_2']))/given_set.shape[0]))

    plt.show()
ARIMA_predict_diff=results_ARIMA.predict(start="2019-03-19 01:00:00", end="2019-03-19 01:33:00")
check_prediction_diff(ARIMA_predict_diff, valid)
import statsmodels.api as sm



# SARIMAX model

y_hat_avg = valid.copy()

SARIMA = sm.tsa.statespace.SARIMAX(df_train.feature_2, order=(1,0,1),seasonal_order=(0,1,1,7))

fit1 = SARIMA.fit()

y_hat_avg['SARIMA'] = fit1.predict(start="2019-03-19 01:00:00", end="2019-03-19 01:33:00", dynamic=True)

plt.figure(figsize=(16,8))

plt.plot( Train['feature_2'], label='Train')

plt.plot(valid['feature_2'], label='Valid')

plt.plot(y_hat_avg['SARIMA'], label='SARIMA')

plt.legend(loc='best')

plt.show()

#calculating rmse value

rmse = sqrt(mean_squared_error(valid.feature_2, y_hat_avg.SARIMA))

print('RMSE value is :',rmse)


predict=fit1.predict(start="2019-03-19 01:34:00", end="2019-03-19 02:36:20", dynamic=True)
df_test['feature_2']=predict
df_test.tail()
submission = df_test.drop(['time','feature_1','Datetime','year','month','day','Hour','seconds','minute'],axis=1)
ids = df_test['id']

feature_2 = df_test['feature_2']
# set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({'Id' : ids, 'Feature_2' : feature_2})

output.to_csv('Submission.csv', index=False)