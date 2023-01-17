import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load the packages

import pandas as pd          

import numpy as np          # For mathematical calculations 

import matplotlib.pyplot as plt  # For plotting graphs 

from datetime import datetime    # To access datetime 

from pandas import Series        # To work on series 

%matplotlib inline 

import warnings                   # To ignore the warnings 

warnings.filterwarnings("ignore")
# Read the data and store the original as a copy:



train=pd.read_csv("../input/Train.csv") 

test=pd.read_csv("../input/Test.csv")



train_original=train.copy() 

test_original=test.copy()
train.columns, test.columns
train.dtypes, test.dtypes
train.shape, test.shape
train.head(), test.head()
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 

test_original['Datetime'] = pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M') 

train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')
#Split the data based on the year, month, date and time

for i in (train, test, test_original, train_original):

    i['year']=i.Datetime.dt.year 

    i['month']=i.Datetime.dt.month 

    i['day']=i.Datetime.dt.day

    i['Hour']=i.Datetime.dt.hour 
train['day of week']=train['Datetime'].dt.dayofweek 

temp = train['Datetime']
#Define the weekends

def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0 

temp2 = train['Datetime'].apply(applyer) 

train['weekend']=temp2
train.index = train['Datetime'] # indexing the Datetime to get the time period on the x-axis. 

df=train.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 

ts = df['Count'] 

plt.figure(figsize=(16,8)) 

plt.plot(ts, label='Passenger Count') 

plt.title('Time Series') 

plt.xlabel("Time(year-month)") 

plt.ylabel("Passenger count") 

plt.legend(loc='best')
# Plotting traffic over the years

train.groupby('year')['Count'].mean().plot.bar()
# Plotting traffic over the average months

train.groupby('month')['Count'].mean().plot.bar()
temp=train.groupby(['year', 'month'])['Count'].mean() 

temp.plot(figsize=(16,5), title= 'Passenger Count(Monthwise)', fontsize=14)
train.groupby('day')['Count'].mean().plot.bar()
train.groupby('Hour')['Count'].mean().plot.bar()
train.groupby('weekend')['Count'].mean().plot.bar()
train.groupby('day of week')['Count'].mean().plot.bar()
#Drop the ID variable 

train=train.drop('ID',1)
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 

# Hourly time series 

hourly = train.resample('H').mean() 

# Converting to daily mean 

daily = train.resample('D').mean() 

# Converting to weekly mean 

weekly = train.resample('W').mean() 

# Converting to monthly mean 

monthly = train.resample('M').mean()
fig, axs = plt.subplots(4,1) 

hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0]) 

daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1])

weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2]) 

monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3]) 



plt.show()
test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 

test.index = test.Timestamp  
# Converting to daily mean 

test = test.resample('D').mean() 

train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 

# Converting to daily mean 

train = train.resample('D').mean()
Train=train.ix['2012-08-25':'2014-06-24'] 

valid=train.ix['2014-06-25':'2014-09-25']
Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 

valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid') 

plt.xlabel("Datetime") 

plt.ylabel("Passenger count") 

plt.legend(loc='best')

plt.show()
# Naive Approach

dd= np.asarray(Train.Count) 

y_hat = valid.copy() 

y_hat['naive'] = dd[len(dd)-1] 

plt.figure(figsize=(12,8)) 

plt.plot(Train.index, Train['Count'], label='Train') 

plt.plot(valid.index,valid['Count'], label='Valid') 

plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 

plt.legend(loc='best') 

plt.title("Naive Forecast") 

plt.show()
from sklearn.metrics import mean_squared_error 

from math import sqrt 

rms = sqrt(mean_squared_error(valid.Count, y_hat.naive)) 

print(rms)
# Moving Average



y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(50).mean().iloc[-1] # average of last 50 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 

plt.legend(loc='best') 

plt.show()



y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(20).mean().iloc[-1] # average of last 20 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations') 

plt.legend(loc='best') 

plt.show() 



y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(10).mean().iloc[-1] # average of last 10 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 

plt.legend(loc='best') 

plt.show() 

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.moving_avg_forecast)) 

print(rms)
# Simple Exponential Smoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

y_hat_avg = valid.copy() 

fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6,optimized=False) 

y_hat_avg['SES'] = fit2.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['SES'], label='SES') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SES)) 

print(rms)
!pip install statsmodels==0.5.0

import statsmodels.api as sm



sm.tsa.seasonal_decompose(Train.Count).plot() 

result = sm.tsa.stattools.adfuller(train.Count) 

plt.show()
# Holt’s Linear Trend Model

y_hat_avg = valid.copy() 

fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)

y_hat_avg['Holt_linear'] = fit1.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_linear)) 

print(rms)
#Holt’s Linear Trend Model on daily time series

predict=fit1.forecast(len(test))

test['prediction']=predict
# Calculating the hourly ratio of count

train_original['ratio']=train_original['Count']/train_original['Count'].sum() 



# Grouping the hourly ratio 

temp=train_original.groupby(['Hour'])['ratio'].sum() 



# Groupby to csv format 

pd.DataFrame(temp, columns=['Hour','ratio']).to_csv('GROUPby.csv') 



temp2=pd.read_csv("GROUPby.csv") 

temp2=temp2.drop('Hour.1',1) 



# Merge Test and test_original on day, month and year 

merge=pd.merge(test, test_original, on=('day','month', 'year'), how='left') 

merge['Hour']=merge['Hour_y'] 

merge=merge.drop(['year', 'month', 'Datetime','Hour_x','Hour_y'], axis=1) 

# Predicting by merging merge and temp2 

prediction=pd.merge(merge, temp2, on='Hour', how='left') 



# Converting the ratio to the original scale 

prediction['Count']=prediction['prediction']*prediction['ratio']*24 

prediction['ID']=prediction['ID_y']
# Holt winter’s model on daily time series

y_hat_avg = valid.copy() 

fit1 = ExponentialSmoothing(np.asarray(Train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit() 

y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot( Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_Winter))

print(rms)
# Dickey-Fuller test.



from statsmodels.tsa.stattools import adfuller 

def test_stationarity(timeseries):

        #Determing rolling statistics

    rolmean = timeseries.rolling(window=24).mean() # 24 hours on each day

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

test_stationarity(train_original['Count'])
Train_log = np.log(Train['Count']) 

valid_log = np.log(valid['Count'])

moving_avg = Train_log.rolling(24).mean() 

plt.plot(Train_log) 

plt.plot(moving_avg, color = 'red') 

plt.show()
train_log_moving_avg_diff = Train_log - moving_avg
train_log_moving_avg_diff.dropna(inplace = True)

test_stationarity(train_log_moving_avg_diff)
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
# ACF and PACF plot

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
# AR model



from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(Train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model 

results_AR = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(), label='original') 

plt.plot(results_AR.fittedvalues, color='red', label='predictions') 

plt.legend(loc='best') 

plt.show()
AR_predict=results_AR.predict(start="2014-06-25", end="2014-09-25")

AR_predict=AR_predict.cumsum().shift().fillna(0)

AR_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['Count'])[0], index = valid.index) 

AR_predict1=AR_predict1.add(AR_predict,fill_value=0) 

AR_predict = np.exp(AR_predict1)

plt.plot(valid['Count'], label = "Valid") 

plt.plot(AR_predict, color = 'red', label = "Predict") 

plt.legend(loc= 'best') 

plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, valid['Count']))/valid.shape[0]))

plt.show()
# MA model



model = ARIMA(Train_log, order=(0, 1, 2))  # here the p value is zero since it is just the MA model 

results_MA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(), label='original') 

plt.plot(results_MA.fittedvalues, color='red', label='prediction') 

plt.legend(loc='best') 

plt.show()
MA_predict=results_MA.predict(start="2014-06-25", end="2014-09-25") 

MA_predict=MA_predict.cumsum().shift().fillna(0) 

MA_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['Count'])[0], index = valid.index) 

MA_predict1=MA_predict1.add(MA_predict,fill_value=0) 

MA_predict = np.exp(MA_predict1)

plt.plot(valid['Count'], label = "Valid") 

plt.plot(MA_predict, color = 'red', label = "Predict") 

plt.legend(loc= 'best') 

plt.title('RMSE: %.4f'% (np.sqrt(np.dot(MA_predict, valid['Count']))/valid.shape[0])) 

plt.show()
# Combined model



model = ARIMA(Train_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(),  label='original') 

plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted') 

plt.legend(loc='best') 

plt.show()
def check_prediction_diff(predict_diff, given_set):

    predict_diff= predict_diff.cumsum().shift().fillna(0)

    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['Count'])[0], index = given_set.index)

    predict_log = predict_base.add(predict_diff,fill_value=0)

    predict = np.exp(predict_log)



    plt.plot(given_set['Count'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))

    plt.show()



def check_prediction_log(predict_log, given_set):

    predict = np.exp(predict_log)

 

    plt.plot(given_set['Count'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))

    plt.show()
ARIMA_predict_diff=results_ARIMA.predict(start="2014-06-25", end="2014-09-25")

check_prediction_diff(ARIMA_predict_diff, valid)
# SARIMAX Model 

import statsmodels.api as sm

y_hat_avg = valid.copy() 

fit1 = sm.tsa.statespace.SARIMAX(Train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit() 

y_hat_avg['SARIMA'] = fit1.predict(start="2014-6-25", end="2014-9-25", dynamic=True) 

plt.figure(figsize=(16,8)) 

plt.plot( Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['SARIMA'], label='SARIMA') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SARIMA)) 

print(rms)
predict=fit1.predict(start="2014-9-26", end="2015-4-26", dynamic=True)
test['prediction']=predict

# Merge Test and test_original on day, month and year 

merge=pd.merge(test, test_original, on=('day','month', 'year'), how='left')

merge['Hour']=merge['Hour_y'] 

merge=merge.drop(['year', 'month', 'Datetime','Hour_x','Hour_y'], axis=1) 



# Predicting by merging merge and temp2 

prediction=pd.merge(merge, temp2, on='Hour', how='left') 



# Converting the ratio to the original scale 

prediction['Count']=prediction['prediction']*prediction['ratio']*24