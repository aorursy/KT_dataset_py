import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from fbprophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use('fivethirtyeight')  

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/Train_SU63ISt.csv')

test = pd.read_csv('/kaggle/input/Test_0qrQsBZ.csv')

sample_sub = pd.read_csv('/kaggle/input/sample_submission_LSeus50.csv')



train_original = train.copy() 

test_original = test.copy()
train.columns, test.columns
train.dtypes, test.dtypes
train.shape, test.shape, sample_sub.shape
train.head(2)
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 



test_original['Datetime'] = pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M') 

train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')
for i in (train, test, test_original, train_original):

    i['year']=i.Datetime.dt.year 

    i['month']=i.Datetime.dt.month 

    i['day']=i.Datetime.dt.day

    i['Hour']=i.Datetime.dt.hour 
train['day of week']=train['Datetime'].dt.dayofweek 

temp = train['Datetime']
# Let’s assign 1 if the day of week is a weekend and 0 if the day of week in not a weekend.



def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0 

temp2 = train['Datetime'].apply(applyer) 

train['weekend']=temp2
#Let’s look at the time series.

train.index = train['Datetime'] # indexing the Datetime to get the time period on the x-axis. 

df=train.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 

ts = df['Count'] 

plt.figure(figsize=(16,8)) 

plt.plot(ts, label='Passenger Count') 

plt.title('Time Series') 

plt.xlabel("Time(year-month)") 

plt.ylabel("Passenger count") 

plt.legend(loc='best')
train.groupby('year')['Count'].mean().plot.bar()
train.groupby('month')['Count'].mean().plot.bar()
temp=train.groupby(['year', 'month'])['Count'].mean() 

temp.plot(figsize=(15,5), title= 'Passenger Count(Monthwise)', fontsize=14)
#Let’s look at the daily mean of passenger count.



train.groupby('day')['Count'].mean().plot.bar()
train.groupby('Hour')['Count'].mean().plot.bar()
train.groupby('weekend')['Count'].mean().plot.bar()
train.groupby('day of week')['Count'].mean().plot.bar()
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
#Let’s look at the hourly, daily, weekly and monthly time series.



pd.plotting.register_matplotlib_converters()



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
Train=train.loc['2012-08-25':'2014-06-24'] 

valid=train.loc['2014-06-25':'2014-09-25']
Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 

valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid') 

plt.xlabel("Datetime") 

plt.ylabel("Passenger count") 

plt.legend(loc='best') 

plt.show()
#Lets try the rolling mean for last 10, 20, 50 days and visualize the results.



y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(10).mean().iloc[-1] # average of last 10 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 

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

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(50).mean().iloc[-1] # average of last 50 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 

plt.legend(loc='best') 

plt.show()
from math import sqrt

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.moving_avg_forecast)) 

print(rms)
# Here the predictions are made by assigning larger weight to the recent values and lesser weight to the old values.



from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

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
# Lets visualize all these parts.



import statsmodels.api as sm 

sm.tsa.seasonal_decompose(Train.Count).plot() 

result = sm.tsa.stattools.adfuller(train.Count) 

plt.figure(figsize=(16,8)) 

plt.show()
# An increasing trend can be seen in the dataset, so now we will make a model based on the trend.



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
submission=pd.read_csv("/kaggle/input/sample_submission_LSeus50.csv")
predict = fit1.forecast(len(test))
# Let’s save these predictions in test file in a new column.

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
# Let’s drop all other features from the submission file and keep ID and Count only.



submission=prediction.drop(['ID_x', 'day', 'ID_y','prediction','Hour', 'ratio'],axis=1) 



# Converting the final submission to csv format 

pd.DataFrame(submission, columns=['ID','Count']).to_csv('Holt linear.csv')
#Let’s first fit the model on training dataset and validate it using the validation dataset.

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
predict=fit1.forecast(len(test))
test['prediction']=predict



# Merge Test and test_original on day, month and year 

merge=pd.merge(test, test_original, on=('day','month', 'year'), how='left') 

merge['Hour']=merge['Hour_y'] 

merge=merge.drop(['year', 'month', 'Datetime','Hour_x','Hour_y'], axis=1) 



# Predicting by merging merge and temp2 

prediction=pd.merge(merge, temp2, on='Hour', how='left') 



# Converting the ratio to the original scale 

prediction['Count']=prediction['prediction']*prediction['ratio']*24
# Let’s drop all features other than ID and Count



prediction['ID']=prediction['ID_y'] 

submission=prediction.drop(['day','Hour','ratio','prediction', 'ID_x', 'ID_y'],axis=1) 



# Converting the final submission to csv format 

pd.DataFrame(submission, columns=['ID','Count']).to_csv('Holt winters.csv')
#Let’s make a function which we can use to calculate the results of Dickey-Fuller test.



from statsmodels.tsa.stattools import adfuller 



def test_stationarity(timeseries):    

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=24).mean()

    rolstd = timeseries.rolling(window=24).std()

    #rolmean = pd.rolling_mean(timeseries, window=24) # 24 hours on each day

    #rolstd = pd.rolling_std(timeseries, window=24)

    

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

#moving_avg = pd.rolling_mean(Train_log, 24) 

moving_avg = Train_log.rolling(window=24).mean()

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
model = ARIMA(Train_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(),  label='original') 

plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted') 

plt.legend(loc='best') 

plt.show()
#Let’s define a function which can be used to change the scale of the model to the original scale.



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
#Let’s predict the values for validation set.



ARIMA_predict_diff=results_ARIMA.predict(start="2014-06-25", end="2014-09-25")

check_prediction_diff(ARIMA_predict_diff, valid)
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
#Let’s check the rmse value for the validation part.



rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SARIMA)) 

print(rms)
# Now we will forecast the time series for Test data which starts from 2014-9-26 and ends at 2015-4-26.

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



# Let’s drop all variables other than ID and Count

prediction['ID']=prediction['ID_y'] 

submission=prediction.drop(['day','Hour','ratio','prediction', 'ID_x', 'ID_y'],axis=1) 



# Converting the final submission to csv format 

pd.DataFrame(submission, columns=['ID','Count']).to_csv('SARIMAX.csv')