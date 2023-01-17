import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("../input/time-series-forecasting/Train_SU63ISt.csv")
train.head(5)
test=pd.read_csv("../input/time-series-forecasting/Test_0qrQsBZ.csv")
test.head()
train.shape
test.shape
train_original=train.copy() 
test_original=test.copy()
train.Datetime=pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index=train.Datetime
train.head(5)
train_original.Datetime=pd.to_datetime(train_original.Datetime, format='%d-%m-%Y %H:%M')
test_original.Datetime=pd.to_datetime(test_original.Datetime, format='%d-%m-%Y %H:%M')
test.Datetime=pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
test.index=test.Datetime
test.head(5)
for i in (train, test, train_original, test_original):
    i['year']=i.Datetime.dt.year 
    i['month']=i.Datetime.dt.month 
    i['day']=i.Datetime.dt.day
    i['Hour']=i.Datetime.dt.hour
train['day of week']=train['Datetime'].dt.dayofweek 
temp = train['Datetime']
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0 
temp2 = train['Datetime'].apply(applyer) 
train['weekend']=temp2
train.drop(['ID','Datetime'],axis=1,inplace=True)
ts = train['Count'] 
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
train.groupby('day')['Count'].mean().plot.bar()
train.groupby('Hour')['Count'].mean().plot.bar()
train.groupby('weekend')['Count'].mean().plot.bar()
train.groupby('day of week')['Count'].mean().plot.bar()
# Hourly time series 
hourly = train.resample('H').mean() 
# Converting to daily mean 
daily = train.resample('D').mean() 
# Converting to weekly mean 
weekly = train.resample('W').mean() 
# Converting to monthly mean 
monthly = train.resample('M').mean()

# Let’s look at the hourly, daily, weekly and monthly time series.

fig, axs = plt.subplots(4,1) 
hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0]) 
daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1]) 
weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2]) 
monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3]) 

plt.show()
# # Converting to daily mean 
# test = test.resample('D').mean() 
monthly=train.resample('D').mean()
monthly.head()
monthly.shape
# train.drop(['year','month','day','Hour','day of week', 'weekend'],axis=1,inplace=True)
train.plot(figsize=(20, 4))
plt.legend(loc='best')
plt.title('Jetrail traffic on hourly basis')
plt.show(block=False)
monthly.plot(figsize=(20, 4))
plt.legend(loc='best')
plt.title('Jetrail traffic on monthly basis')
plt.show(block=False)
import seaborn as sns
fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=monthly['Count'],whis=1.5)
fig = monthly.Count.hist(figsize = (12,4))
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(monthly.Count, model='additive') # additive seasonal index
fig = decomposition.plot()
plt.show()
decomposition = sm.tsa.seasonal_decompose(monthly.Count, model='multiplicative') # multiplicative seasonal index
fig = decomposition.plot()
plt.show()
# train_len = 650
# train = monthly[0:train_len] # first 120 months as training set
# test = monthly[train_len:] # last 24 months as out-of-time test set
Train=monthly.loc['2012-08-25':'2014-06-24'] 
valid=monthly.loc['2014-06-25':'2014-09-25']
Train.shape
Train.head()
valid.shape
from statsmodels.tsa.holtwinters import ExponentialSmoothing

y_hat_hwa = valid.copy()
model = ExponentialSmoothing(np.asarray(Train['Count']) ,seasonal_periods=4,trend='add', seasonal='add')
model_fit = model.fit(optimized=True)
print(model_fit.params)
y_hat_hwa['hw_forecast'] = model_fit.forecast(len(valid))
plt.figure(figsize=(12,4))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Test')
plt.plot(y_hat_hwa['hw_forecast'], label='Holt Winters\'s additive forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Additive Method')
plt.show()
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(valid['Count'], y_hat_hwa['hw_forecast'])).round(2)
mape = np.round(np.mean(np.abs(valid['Count']-y_hat_hwa['hw_forecast'])/valid['Count'])*100,2)

tempResults = pd.DataFrame({'Method':['Holt Winters\' additive method'], 'RMSE': [rmse],'MAPE': [mape]})
# results = pd.concat([results, tempResults])
results = tempResults[['Method', 'RMSE', 'MAPE']]
results
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(monthly['Count'])

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])
from scipy.stats import boxcox
data_boxcox = pd.Series(boxcox(monthly['Count'], lmbda=0), index = monthly.index)

plt.figure(figsize=(12,4))
plt.plot(data_boxcox, label='After Box Cox tranformation')
plt.legend(loc='best')
plt.title('After Box Cox transform')
plt.show()
data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), monthly.index)
plt.figure(figsize=(12,4))
plt.plot(data_boxcox_diff, label='After Box Cox tranformation and differencing')
plt.legend(loc='best')
plt.title('After Box Cox transform and differencing')
plt.show()
data_boxcox_diff.dropna(inplace=True)
data_boxcox_diff.tail()
data_boxcox_diff.head()
adf_test = adfuller(data_boxcox_diff)

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(12,4))
plot_acf(data_boxcox_diff, ax=plt.gca(), lags = 20)
plt.show()
from statsmodels.graphics.tsaplots import plot_pacf
plt.figure(figsize=(12,4))
plot_pacf(data_boxcox_diff, ax=plt.gca(), lags = 20)
plt.show()
train_len = 669
train_data_boxcox = data_boxcox[:train_len]
test_data_boxcox = data_boxcox[train_len:]
train_data_boxcox_diff = data_boxcox_diff[:train_len-1]
test_data_boxcox_diff = data_boxcox_diff[train_len-1:]
train_data_boxcox_diff
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(train_data_boxcox_diff, order=(1, 0, 0)) 
model_fit = model.fit()
print(model_fit.params)
y_hat_ar = data_boxcox_diff.copy()
y_hat_ar['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox_diff'].cumsum()
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox'].add(data_boxcox[0])
y_hat_ar['ar_forecast'] = np.exp(y_hat_ar['ar_forecast_boxcox'])
plt.figure(figsize=(12,4))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Test')
plt.plot(y_hat_ar['ar_forecast'][valid.index.min():], label='Auto regression forecast')
plt.legend(loc='best')
plt.title('Auto Regression Method')
plt.show()
rmse = np.sqrt(mean_squared_error(valid['Count'], y_hat_ar['ar_forecast'][valid.index.min():])).round(2)
mape = np.round(np.mean(np.abs(valid['Count']-y_hat_ar['ar_forecast'][valid.index.min():])/valid['Count'])*100,2)

tempResults = pd.DataFrame({'Method':['Autoregressive (AR) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results
model = ARIMA(train_data_boxcox_diff, order=(0, 0, 1)) 
model_fit = model.fit()
print(model_fit.params)
y_hat_ma = data_boxcox_diff.copy()
y_hat_ma['ma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox_diff'].cumsum()
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox'].add(data_boxcox[0])
y_hat_ma['ma_forecast'] = np.exp(y_hat_ma['ma_forecast_boxcox'])

plt.figure(figsize=(12,4))
plt.plot(monthly['Count'][:train_len], label='Train')
plt.plot(monthly['Count'][train_len:], label='Test')
plt.plot(y_hat_ma['ma_forecast'][valid.index.min():], label='Moving average forecast')
plt.legend(loc='best')
plt.title('Moving Average Method')
plt.show()

rmse = np.sqrt(mean_squared_error(valid['Count'], y_hat_ma['ma_forecast'][valid.index.min():])).round(2)
mape = np.round(np.mean(np.abs(valid['Count']-y_hat_ma['ma_forecast'][valid.index.min():])/valid['Count'])*100,2)

tempResults = pd.DataFrame({'Method':['Moving Average (MA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results
model = ARIMA(train_data_boxcox_diff, order=(1, 0, 1))
model_fit = model.fit()
print(model_fit.params)
y_hat_arma = data_boxcox_diff.copy()
y_hat_arma['arma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox_diff'].cumsum()
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox'].add(data_boxcox[0])
y_hat_arma['arma_forecast'] = np.exp(y_hat_arma['arma_forecast_boxcox'])
plt.figure(figsize=(12,4))
plt.plot(monthly['Count'][:train_len-1], label='Train')
plt.plot(monthly['Count'][train_len-1:], label='Test')
plt.plot(y_hat_arma['arma_forecast'][valid.index.min():], label='ARMA forecast')
plt.legend(loc='best')
plt.title('ARMA Method')
plt.show()
rmse = np.sqrt(mean_squared_error(valid['Count'], y_hat_arma['arma_forecast'][train_len-1:])).round(2)
mape = np.round(np.mean(np.abs(valid['Count']-y_hat_arma['arma_forecast'][train_len-1:])/valid['Count'])*100,2)

tempResults = pd.DataFrame({'Method':['Autoregressive moving average (ARMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results
model = ARIMA(train_data_boxcox, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.params)
y_hat_arima = data_boxcox_diff.copy()
y_hat_arima['arima_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox_diff'].cumsum()
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox'].add(data_boxcox[0])
y_hat_arima['arima_forecast'] = np.exp(y_hat_arima['arima_forecast_boxcox'])
plt.figure(figsize=(12,4))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Test')
plt.plot(y_hat_arima['arima_forecast'][valid.index.min():], label='ARIMA forecast')
plt.legend(loc='best')
plt.title('Autoregressive integrated moving average (ARIMA) method')
plt.show()
rmse = np.sqrt(mean_squared_error(valid['Count'], y_hat_arima['arima_forecast'][valid.index.min():])).round(2)
mape = np.round(np.mean(np.abs(valid['Count']-y_hat_arima['arima_forecast'][valid.index.min():])/valid['Count'])*100,2)

tempResults = pd.DataFrame({'Method':['Autoregressive integrated moving average (ARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_data_boxcox, order=(3, 1, 4), seasonal_order=(0, 1, 1, 7)) 
model_fit = model.fit()
print(model_fit.params)
y_hat_sarima = data_boxcox_diff.copy()
y_hat_sarima['sarima_forecast_boxcox'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_sarima['sarima_forecast'] = np.exp(y_hat_sarima['sarima_forecast_boxcox'])
plt.figure(figsize=(12,4))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Test')
plt.plot(y_hat_sarima['sarima_forecast'][valid.index.min():], label='SARIMA forecast')
plt.legend(loc='best')
plt.title('Seasonal autoregressive integrated moving average (SARIMA) method')
plt.show()
rmse = np.sqrt(mean_squared_error(valid['Count'], y_hat_sarima['sarima_forecast'][valid.index.min():])).round(2)
mape = np.round(np.mean(np.abs(valid['Count']-y_hat_sarima['sarima_forecast'][valid.index.min():])/valid['Count'])*100,2)

tempResults = pd.DataFrame({'Method':['Seasonal autoregressive integrated moving average (SARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results
# test.drop(['ID','year','month','day','Hour'],axis=1,inplace=True)
print(len(test))
test.head()
test1=test.copy()
test1.drop(['ID','year','month','day','Hour','Datetime'],axis=1,inplace=True)
test1.head()
predict=model_fit.predict(start="2014-9-26", end="2015-4-26", dynamic=True)
predict=np.exp(predict)
test['prediction']=predict
test.head()
# Remember this is the daily predictions. 
# We have to convert these predictions to hourly basis. 
# To do so we will first calculate the ratio of passenger count for each hour of every day. 
# Then we will find the average ratio of passenger count for every hour and we will get 24 ratios. 
# Then to calculate the hourly predictions we will multiply the daily prediction with the hourly ratio.

# Calculating the hourly ratio of count 
train_original['ratio']=train_original['Count']/train_original['Count'].sum() 


# Grouping the hourly ratio 
temp=train_original.groupby(['Hour'])['ratio'].sum() 
# Groupby to csv format 
pd.DataFrame(temp, columns=['Hour','ratio']).to_csv('GROUPby.csv') 

temp2=pd.read_csv("GROUPby.csv") 
temp2=temp2.drop('Hour.1',1) 
temp2.head()
# Merge Test and test_original on day, month and year 
merge=pd.merge(test, test_original, on=('day','month','year','Hour'), how='left') 

for i in range(0,len(merge)+1):
        merge['prediction'].fillna(method ='pad', inplace=True) 
merge.head(50)
merge=merge.drop(['year', 'month','Datetime_x','Datetime_x','Datetime_y'], axis=1) 

# Predicting by merging merge and temp2 
prediction=pd.merge(merge, temp2, on='Hour', how='left') 
# Converting the ratio to the original scale 
prediction['Count']=prediction['prediction']*prediction['ratio']*24
prediction['ID']=prediction['ID_y'] 
submission=prediction.drop(['day','Hour','ratio','prediction', 'ID_x', 'ID_y'],axis=1) 
# Converting the final submission to csv format 
pd.DataFrame(submission, columns=['ID','Count']).to_csv('SARIMA.csv',index=False)
submission.head()