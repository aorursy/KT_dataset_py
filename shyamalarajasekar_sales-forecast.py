import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import datetime

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
store = pd.read_csv('../input/global-mart-sales-data/Superstore.csv')

store.head()
store['Market-Seg'] = store['Segment']+' - '+store['Market']

#store['Order Date'] = pd.to_datetime(store['Order Date'])

store['Date'] = pd.to_datetime(store['Order Date']).dt.to_period('m')

store.head()
pd.DataFrame(store['Market-Seg'].value_counts(normalize = True)*100)
store.shape
store.info()
store.describe()
store.sort_values(by='Date', inplace = True)

print('Order date starts from', store['Date'].min(),'and ends at',store['Date'].max())
store.head()
plt.figure(figsize = (20,5))

sns.boxplot(x = 'Sales', data = store)

plt.xscale('log')

plt.title('Boxplot of Sales in Global Mart from 2011-14')
plt.figure(figsize = (20,5))

plot = sns.boxplot(x = 'Profit', data = store)

plt.title('Boxplot of Profit in Global Mart from 2011-14')
plt.figure(figsize = (20,4))

plt.title('Market Plot')

sns.countplot(x = 'Market', data = store)
plt.figure(figsize = (20,4))

plt.title('Segment Plot')

sns.countplot(x = 'Segment', data = store)
plt.figure(figsize = (20,4))

plt.title('Market Segment Plot')

sns.countplot(x = 'Market-Seg', data = store)

plt.xticks(rotation = 45)
plt.figure(figsize = (15,5))

plt.xticks(rotation = 90)

plt.title('Market Segment Vs Sales')

sns.barplot(data = store, x = 'Market-Seg', y = 'Sales')
plt.figure(figsize = (15,5))

plt.xticks(rotation = 90)

sns.barplot(data = store, x = 'Market-Seg', y = 'Profit')
store.head()
storeg = store.pivot_table(index='Date', values='Profit', columns='Market-Seg', aggfunc='sum')
train_len = 42

train = storeg[0:train_len]

test = storeg[train_len:]
mean = np.mean(train)

std = np.std(train)

CoV = std/mean

CoV = pd.DataFrame(CoV)

CoV = CoV.reset_index()

CoV.columns = ['Market segment','Most Consistent']

CoV.sort_values(by = 'Most Consistent', ascending = True, inplace = True)

CoV
df = store[store['Market-Seg'] == 'Consumer - APAC']

df = df.drop('Order Date', axis = 1)

df.head()
df.shape
df.describe()
df['Date'] = df['Date'].astype(str)

plt.figure(figsize = (20,4))

sns.lineplot(x = 'Date',y = 'Profit', data = df)

plt.title('Profit Between 2011 to 2014')

plt.xticks(rotation = 45)

plt.show()
df = df.groupby('Date')['Sales'].sum()



### new dataframe df sales in a particular date are summed together to find the total sum of sales on that day.



df = pd.DataFrame(df) ## converting the series into a dataframe

df = df.reset_index()



df['Date'] = pd.to_datetime(df['Date'] ) ## converting the date into datetime format

df = df.set_index('Date')

df = df.sort_values(by = 'Date', ascending = True) ## sorting the values by date for data so that test set lies after train set
df.shape
df.info()
df.plot(figsize=(25, 4))

plt.legend(loc='best')

plt.grid(color='b', linestyle='-', linewidth=0.1)

plt.title('Global Sales')

plt.show(block=False)
from pylab import rcParams

import statsmodels.api as sm

rcParams['figure.figsize'] = 12, 8

decomposition = sm.tsa.seasonal_decompose(df.Sales, model='additive') # additive seasonal index

fig = decomposition.plot()

plt.show()
decomposition = sm.tsa.seasonal_decompose(df.Sales, model='multiplicative') # multiplicative seasonal index

fig = decomposition.plot()

plt.show()
train_length = 42

trainset = df[0:train_length] # 42 months 

testset = df[train_length:] # 6 months
y_hat_naive = testset.copy()

y_hat_naive['Naive forecast'] = trainset['Sales'][train_length-1] # last value is taken as the forecast. 
def forecastplot(x, y):

    plt.figure(figsize = (20,4))

    plt.plot(trainset['Sales'], label = 'Train')

    plt.plot(testset['Sales'], label = 'Test')

    plt.plot(x, label = y )

    plt.title(y)

    plt.legend(loc='best')

    plt.xticks(rotation = 45)

    plt.show()
forecastplot(y_hat_naive['Naive forecast'], 'Naive forecast')
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_naive['Naive forecast'])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_naive['Naive forecast'])/testset['Sales'])*100,2)



results = pd.DataFrame({'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_avg = testset.copy()

y_hat_avg['Average forecast'] = trainset['Sales'].mean()
forecastplot(y_hat_avg['Average forecast'], 'Simple Average forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_avg['Average forecast'])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_avg['Average forecast'])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple Average method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_sma = df.copy()

ma_window12 = 12 ## 12 months 

y_hat_sma['sma_forecast12'] = df['Sales'].rolling(ma_window12).mean()

y_hat_sma['sma_forecast12'][train_length:] = y_hat_sma['sma_forecast12'][train_length-1]
forecastplot(y_hat_sma['sma_forecast12'][train_length:], 'Simple Moving Average Forecast for 12 Months')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_sma['sma_forecast12'][train_length:])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_sma['sma_forecast12'][train_length:])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple Moving Average Forecast (12 Months)'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_sma = df.copy()

ma_window6 = 6

y_hat_sma['sma_forecast6'] = df['Sales'].rolling(ma_window6).mean()

y_hat_sma['sma_forecast6'][train_length:] = y_hat_sma['sma_forecast6'][train_length-1]
forecastplot(y_hat_sma['sma_forecast6'][train_length:], 'Simple Moving Average Forecast for 6 Months')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_sma['sma_forecast6'][train_length:])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_sma['sma_forecast6'][train_length:])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple Moving Average Forecast (6 Months)'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_sma = df.copy()

ma_window3 = 3

y_hat_sma['sma_forecast3'] = df['Sales'].rolling(ma_window3).mean()

y_hat_sma['sma_forecast3'][train_length:] = y_hat_sma['sma_forecast3'][train_length-1]
forecastplot(y_hat_sma['sma_forecast3'][train_length:], 'Simple Moving Average Forecast for 3 Months')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_sma['sma_forecast3'][train_length:])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_sma['sma_forecast3'][train_length:])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple Moving Average Forecast (3 Months)'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(trainset['Sales'])

model_fit = model.fit(optimized=True) ## the most optimal alpha parameter will be chosen 

print(model_fit.params)

y_hat_ses = testset.copy()

y_hat_ses['ses_forecast'] = model_fit.forecast(len(testset))
forecastplot(y_hat_ses['ses_forecast'], 'Simple Exponential Smoothing Forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_ses['ses_forecast'])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_ses['ses_forecast'])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple Exponential method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(np.asarray(trainset['Sales']) ,seasonal_periods = 12 ,trend='additive', seasonal=None)

model_fit = model.fit(optimized=True) # most optimum alpha and beta values are taken

print(model_fit.params)

y_hat_holt = testset.copy()

y_hat_holt['holt_forecast'] = model_fit.forecast(len(test))
forecastplot(y_hat_holt['holt_forecast'], 'Holt\'s exponential smoothing forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_holt['holt_forecast'])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_holt['holt_forecast'])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt Exp method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_hwa = testset.copy()

model = ExponentialSmoothing(np.asarray(trainset['Sales']) ,seasonal_periods=12 ,trend='add', seasonal='add')

model_fit = model.fit(optimized=True)

print(model_fit.params)

y_hat_hwa['hwa_forecast'] = model_fit.forecast(len(testset))
forecastplot(y_hat_hwa['hwa_forecast'], 'Holt\'s exponential smoothing forecast with Trend and Season')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_hwa['hwa_forecast'])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_hwa['hwa_forecast'])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt Exponential Additive method '], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_hwm = testset.copy()

model = ExponentialSmoothing(np.asarray(trainset['Sales']) ,seasonal_periods=12 ,trend='add', seasonal='mul')

model_fit = model.fit(optimized=True)

print(model_fit.params)

y_hat_hwm['hwm_forecast'] = model_fit.forecast(len(testset))
forecastplot(y_hat_hwm['hwm_forecast'], 'Holt\'s Exponential Multiplicative smoothing forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_hwm['hwm_forecast'])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_hwm['hwm_forecast'])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt Exponential method Multiplicative'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
plt.figure(figsize=(25,10))

plt.plot(trainset['Sales'], label='Train')

plt.plot(testset['Sales'], label='Test')

plt.plot(y_hat_avg['Average forecast'], label='Simple average forecast',linewidth = 3)

plt.plot(y_hat_naive['Naive forecast'], label = 'Naive Forecast',linewidth = 3, color = 'pink')

plt.plot(y_hat_sma['sma_forecast3'][train_length:], label = 'Simple Moving Average Forecast for 3 Months',linewidth = 1, color = 'red')

plt.plot(y_hat_hwa['hwa_forecast'], label = 'Holt\'s Additive Exponential Smoothing Technique',linewidth = 3)

plt.plot(y_hat_hwm['hwm_forecast'], label = 'Holt\'s Multiplicative Exponential Smoothing Technique',linewidth = 3)

plt.plot(y_hat_ses['ses_forecast'], label = 'Simple Exponential Smoothing Technique',linewidth = 3,linestyle = 'dotted')

plt.legend(loc='best')

plt.title('All Smoothing Forecasts')

plt.show()
plt.figure(figsize = (20,4))

plt.title('Plot of the Sales of Global Mart')

df['Sales'].plot()
from statsmodels.tsa.stattools import kpss

kpss_test = kpss(df['Sales'])

print(kpss_test)



print('KPSS Statistic: %f' % kpss_test[0])

print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])

print('p-value: %f' % kpss_test[1])
from scipy.stats import boxcox

df_boxcox = pd.Series(boxcox(df['Sales'], lmbda=0), index = df.index) ## creating a new series called df_boxcox



plt.figure(figsize=(20,4))

plt.plot(df_boxcox, label='After Box Cox tranformation')

plt.legend(loc='best')

plt.title('After Box Cox transform')

plt.show()
df_boxcox_diff = pd.Series(df_boxcox - df_boxcox.shift(), df.index)

## shift operator gives the consequetive difference i.e. difference between 2 consequetive terms

df_boxcox_diff.dropna(inplace = True) ## first difference would be between 2nd and 1st. the first term will be null. 



plt.figure(figsize=(20,4))

plt.plot(df_boxcox_diff, label='After Box Cox tranformation and differencing')

plt.legend(loc='best')

plt.title('After Box Cox transform and differencing')

plt.show()
## splitting both data after bocox and data after boxcox and differencing

train_data_boxcox = df_boxcox[:train_length]

test_data_boxcox = df_boxcox[train_length:]



train_data_boxcox_diff = df_boxcox_diff[:train_length-1]

test_data_boxcox_diff = df_boxcox_diff[train_length-1:]
from statsmodels.tsa.arima_model import ARIMA # I and M and A parameters are closed. 

model = ARIMA(train_data_boxcox_diff, order=(1, 0, 0)) ### Only the first parameter is given and it is lag here. 

# We are giving a lag of 1 here i.e. we have only one y(t-1)

model_fit = model.fit()

print(model_fit.params)
y_hat_ar = df_boxcox_diff.copy()

y_hat_ar['ar_forecast_boxcox_diff'] = model_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max()) 

# entire data range

y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox_diff'].cumsum() 

## doing the opposite of differencing : cumulative sum

y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox'].add(df_boxcox[0])

y_hat_ar['ar_forecast'] = np.exp(y_hat_ar['ar_forecast_boxcox']) # converting log to anti log is to raise it to an exponential
forecastplot(y_hat_ar['ar_forecast'],'Auto regression forecast with Trainset Forecast')
forecastplot(y_hat_ar['ar_forecast'][testset.index.min():],'Auto regression forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_ar['ar_forecast'][testset.index.min():])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_ar['ar_forecast'][testset.index.min():])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Autoregressive (AR) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
model = ARIMA(train_data_boxcox_diff, order=(0, 0, 1)) ## q is 1

model_fit = model.fit()

print(model_fit.params)
y_hat_ma = df_boxcox_diff.copy()

y_hat_ma['ma_forecast_boxcox_diff'] = model_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max())

y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox_diff'].cumsum() 

## taking the cumulative sum to reverse differencing 

y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox'].add(df_boxcox[0])

y_hat_ma['ma_forecast'] = np.exp(y_hat_ma['ma_forecast_boxcox'])

## reversing transformation

y_hat_ma.head()
forecastplot(y_hat_ma['ma_forecast'], 'Moving average forecast with Trainset Forecast')
forecastplot(y_hat_ma['ma_forecast'][testset.index.min():], 'Moving average forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_ma['ma_forecast'][testset.index.min():])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_ma['ma_forecast'][testset.index.min():])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Moving Average (MA) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
model = ARIMA(train_data_boxcox_diff, order=(1, 0, 1))

## p = 1 and q = 1

model_fit = model.fit()

print(model_fit.params)
y_hat_arma = df_boxcox_diff.copy()

y_hat_arma['arma_forecast_boxcox_diff'] = model_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max())

### predicting for the whole year -> data_boxcox_diff.index.min(), data_boxcox_diff.index.max()



y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox_diff'].cumsum()

y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox'].add(df_boxcox[0])

y_hat_arma['arma_forecast'] = np.exp(y_hat_arma['arma_forecast_boxcox'])
forecastplot(y_hat_arma['arma_forecast'], 'ARMA Forecast with Trainset forecast')
forecastplot(y_hat_arma['arma_forecast'][testset.index.min():], 'ARMA Forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_arma['arma_forecast'][train_length-1:])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_arma['arma_forecast'][train_length-1:])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Autoregressive moving average (ARMA) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
model = ARIMA(train_data_boxcox, order=(1, 1, 1))

## we dont have to differentiate data, ARIMA does it on its own. So only boxcox transformed data is used here. 



model_fit = model.fit()

print(model_fit.params)
y_hat_arima = df_boxcox_diff.copy()

## since d is 1, we will get a differenced series as an output. Hence we pick data_boxcox_diff 

y_hat_arima['arima_forecast_boxcox_diff'] = model_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max())

y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox_diff'].cumsum()

y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox'].add(df_boxcox[0])

y_hat_arima['arima_forecast'] = np.exp(y_hat_arima['arima_forecast_boxcox'])
forecastplot(y_hat_arima['arima_forecast'], 'ARIMA forecast with Trainset Forecast')
forecastplot(y_hat_arima['arima_forecast'][testset.index.min():], 'ARIMA forecast with Trainset Forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_arima['arima_forecast'][testset.index.min():])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_arima['arima_forecast'][testset.index.min():])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Autoregressive integrated moving average (ARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
from statsmodels.tsa.statespace.sarimax import SARIMAX



model = SARIMAX(train_data_boxcox, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) 

## Non-seasonal elements p =1, q=1,d =1

## seasonal elements P = 1, Q= 1, D = 1, m= 12



## order is for non-seasonal parameters, seasonal orders are P,Q,D,m

model_fit = model.fit()

print(model_fit.params)
y_hat_sarima = df_boxcox_diff.copy()

y_hat_sarima['sarima_forecast_boxcox'] = model_fit.predict(df_boxcox_diff.index.min(), df_boxcox_diff.index.max())

## We dont have to the cumulative sum as we used to do in ARIMA. 

##  We dont have to do the integration part as it is directly done in SARIMA



y_hat_sarima['sarima_forecast'] = np.exp(y_hat_sarima['sarima_forecast_boxcox'])
forecastplot(y_hat_sarima['sarima_forecast'][testset.index.min():], 'SARIMA Forecast')
rmse = np.sqrt(mean_squared_error(testset['Sales'], y_hat_sarima['sarima_forecast'][testset.index.min():])).round(2)

mape = np.round(np.mean(np.abs(testset['Sales']-y_hat_sarima['sarima_forecast'][testset.index.min():])/testset['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Seasonal autoregressive integrated moving average (SARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results

results.set_index('Method')
plt.figure(figsize=(25,10))

plt.plot(trainset['Sales'], label='Train')

plt.plot(testset['Sales'], label='Test')



plt.plot(y_hat_ar['ar_forecast'][testset.index.min():], label = 'AR Forecast', linewidth = 3)

plt.plot(y_hat_ma['ma_forecast'][testset.index.min():], label = 'Moving Average Forecast',linewidth = 3)

plt.plot(y_hat_arma['arma_forecast'][testset.index.min():], label = 'ARMA',linewidth = 3)

plt.plot(y_hat_arima['arima_forecast'][testset.index.min():], label = 'ARIMA',linewidth = 3)

plt.plot(y_hat_sarima['sarima_forecast'][testset.index.min():], label = 'SARIMA',linewidth = 3)

plt.legend(loc='best')

plt.title('All Autoregression Forecasts')

plt.show()