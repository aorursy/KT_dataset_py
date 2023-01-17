# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn; seaborn.set()

from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15,6

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.arima_model import ARIMA
data = pd.read_csv("../input/bitcoin_price_Training - Training.csv")

print(data.head(5))

print(data.tail(5))

data.dtypes

data.info()

data.describe()
data = pd.read_csv("../input/bitcoin_price_Training - Training.csv",index_col= 'Date')

print(data.head(5)) 

data.info()

data.index = pd.to_datetime(data.index)

print(data.index)

data.head(5)
data = data.sort_index()

data.head()
data['Close'].plot()

plt.ylabel("DAily Bitcoin price")
data = data['Close']
weekly = data.resample('W').sum()

weekly.plot()

plt.ylabel('Weekly bitcoin price')

by_year = data.groupby(data.index.year).mean()

by_year.plot()
by_weekday = data.groupby(data.index.dayofweek).sum()

by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

by_weekday.plot()
by_weekday = data.groupby(data.index.dayofweek).mean()

by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

by_weekday.plot()
by_day = data.groupby(data.index.dayofyear).mean()

by_day.plot()
by_month = data.groupby(data.index.month).mean()

by_month.plot()
by_quarter = data.groupby(data.index.quarter).mean()

by_quarter.plot()
by_quarter = data.groupby(data.index.quarter)

by_quarter.plot()
by_quarter_overall = data.groupby(data.index.quarter).mean()

by_quarter_overall.plot()
weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')

by_time = data.groupby([weekend, data.index.year]).mean()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays')

by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends')
ts = data
plt.plot(ts)
from statsmodels.tsa.stattools import adfuller

def test_for_stationary(input_data):

    r_mean = input_data.rolling(window = 7,center=False).mean()

    r_std = input_data.rolling(window = 7,center=False).std()

    

    # plotting the data

    given = plt.plot(input_data, color = 'blue', label = 'given_series')

    rolling_mean = plt.plot(r_mean, color = 'red', label = 'rolling_mean')

    rolling_std = plt.plot(r_std, color ='green', label = 'rolling_std')

    plt.legend(loc = 'best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

     #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(input_data)

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

      

test_for_stationary(ts)   
ts_logtransformed = np.log(ts)

plt.plot(ts_logtransformed)
ts_logtransformed.head(10)
Rolling_average = ts_logtransformed.rolling(window = 7, center= False).mean()

plt.plot(ts_logtransformed, label = 'Log Transformed')

plt.plot(Rolling_average, color = 'red', label = 'Rolling Average')

plt.legend(loc = 'best')

Rolling_average.head(10)
log_Rolling_difference = ts_logtransformed - Rolling_average

log_Rolling_difference.head(10)

log_Rolling_difference.tail(10)
log_Rolling_difference.dropna(inplace=True)

plt.plot(log_Rolling_difference)
test_for_stationary(log_Rolling_difference)
expwighted_avg = ts_logtransformed.ewm(halflife=7,min_periods=0,adjust=True,ignore_na=False).mean()

plt.plot(ts_logtransformed, label = 'Log transfomed')

plt.plot(expwighted_avg, color='red', label = 'exponential weighted average')

plt.legend(loc = 'best')
expwighted_avg.head(10)
log_expmovwt_diff = ts_logtransformed - expwighted_avg



test_for_stationary(log_expmovwt_diff)
ts_logtransformed.plot()
#X = ts_logtransformed

#diff = list()

#days_in_quarter = 91

#for i in range(days_in_quarter, len(X)):

 #   value = X[i] - X[i - days_in_quarter]

#    diff.append(value)

#plt.plot(diff)

#diff = pd.Series(diff)

#diff

#plt.plot(diff)
ts_diff_logtrans = ts_logtransformed -ts_logtransformed.shift(7)

plt.plot(ts_diff_logtrans)

ts_diff_logtrans.head(10)
ts_diff_logtrans.dropna(inplace=True)

test_for_stationary(ts_diff_logtrans)
decomposition = seasonal_decompose(ts_logtransformed)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(ts_logtransformed, label='Original')

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
decomposed_TS = residual

decomposed_TS.dropna(inplace=True)

test_for_stationary(decomposed_TS)

#ACF and PACF plots:

lag_acf = acf(ts_diff_logtrans, nlags=30)

lag_pacf = pacf(ts_diff_logtrans, nlags=50, method='ols')
#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')
#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

from matplotlib import pyplot

pyplot.figure()

pyplot.subplot(211)

plot_acf(ts_diff_logtrans, ax=pyplot.gca(),lags=40)

pyplot.subplot(212)

plot_pacf(ts_diff_logtrans, ax=pyplot.gca(), lags=50)

pyplot.show()
from statsmodels.tsa.arima_model import ARIMA
ts_diff_logtrans = ts_diff_logtrans.fillna(0)
#model = ARIMA(ts_logtransformed, order=(20, 1, 0))  

#results_AR = model.fit(disp=-1)  

#plt.plot(ts_diff_logtrans)

#plt.plot(results_AR.fittedvalues, color='red', label = 'order 20')

#RSS = results_AR.fittedvalues-ts_diff_logtrans

#RSS.dropna(inplace=True)

#plt.title('RSS: %.4f'% sum(RSS**2))

#plt.legend(loc = 'best')
#model = ARIMA(ts_logtransformed, order=(15, 1, 0))  

#results_AR = model.fit(disp=-1)  

#plt.plot(ts_diff_logtrans)

#plt.plot(results_AR.fittedvalues, color='red', label = 'order 15')

#RSS = results_AR.fittedvalues-ts_diff_logtrans

#RSS.dropna(inplace=True)

#plt.title('RSS: %.4f'% sum(RSS**2))

#plt.legend(loc = 'best')
#model = ARIMA(ts_logtransformed, order=(12, 1, 0))  

#results_AR = model.fit(disp=-1)  

#plt.plot(ts_diff_logtrans)

#plt.plot(results_AR.fittedvalues, color='red', label = 'order 12')

#RSS = results_AR.fittedvalues-ts_diff_logtrans

#RSS.dropna(inplace=True)

#plt.title('RSS: %.4f'% sum(RSS**2))

#plt.legend(loc = 'best')
#model = ARIMA(ts_logtransformed, order=(7, 1, 0))  

#results_AR = model.fit(disp=-1)  

#plt.plot(ts_diff_logtrans)

#plt.plot(results_AR.fittedvalues, color='red', label = 'order 7')

#RSS = results_AR.fittedvalues-ts_diff_logtrans

#RSS.dropna(inplace=True)

#plt.title('RSS: %.4f'% sum(RSS**2))

#plt.legend(loc = 'best')
model = ARIMA(ts_logtransformed, order=(8, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.plot(ts_diff_logtrans)

plt.plot(results_AR.fittedvalues, color='red', label = 'order 8')

RSS = results_AR.fittedvalues-ts_diff_logtrans

RSS.dropna(inplace=True)

plt.title('RSS: %.4f'% sum(RSS**2))

plt.legend(loc = 'best')
model = ARIMA(ts_logtransformed, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.plot(ts_diff_logtrans)

plt.plot(results_AR.fittedvalues, color='red', label = 'order 2')

RSS = results_AR.fittedvalues-ts_diff_logtrans

RSS.dropna(inplace=True)

plt.title('RSS: %.4f'% sum(RSS**2))

plt.legend(loc ='best')



print(results_AR.summary())
model = ARIMA(ts_logtransformed, order=(0, 1,18)) 

results_MA = model.fit(disp=-1)  

plt.plot(ts_diff_logtrans)

plt.plot(results_MA.fittedvalues, color='red')

RSS = results_MA.fittedvalues-ts_diff_logtrans

RSS.dropna(inplace=True)

plt.title('RSS: %.4f'% sum(RSS**2))

#model summary

print(results_MA.summary())

plt.plot(ts_logtransformed, label = 'log_tranfromed_data')

plt.plot(results_MA.resid, color ='green',label= 'Residuals')

plt.title('MA Model Residual plot')

plt.legend(loc = 'best')
results_MA.resid.plot(kind='kde')

plt.title('Density plot of the residual error values')

print(results_MA.resid.describe())
model = ARIMA(ts_logtransformed, order=(8, 1, 18))  

results_ARIMA = model.fit(trend= 'nc', disp=-1)  

plt.plot(ts_diff_logtrans)

plt.plot(results_ARIMA.fittedvalues, color='red', label = 'p =8, q =18')

RSS =results_ARIMA.fittedvalues-ts_diff_logtrans

RSS.dropna(inplace=True)

plt.title('RSS: %.4f'% sum(RSS**2))

plt.legend(loc='best')
#model summary

print(results_ARIMA.summary())
plt.plot(ts_logtransformed, label = 'log_tranfromed_data')

plt.plot(results_ARIMA.resid, color ='green',label= 'Residuals')

plt.title('ARIMA Model Residual plot')

plt.legend(loc = 'best')
results_ARIMA.resid.plot(kind='kde')

plt.title('Density plot of the residual error values')

print(results_ARIMA.resid.describe())
test = pd.read_csv("../input/bitcoin_price_1week_Test - Test.csv",index_col= 'Date')

test.index = pd.to_datetime(test.index)

test = test['Close']

test = test.sort_index()

test
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(ts_logtransformed.iloc[0], index=ts_logtransformed.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(data)

plt.plot(predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-data)**2)/len(data)))
dates = [pd.Timestamp('2017-08-01'), pd.Timestamp('2017-08-02'), pd.Timestamp('2017-08-03'),pd.Timestamp('2017-08-04'), pd.Timestamp('2017-08-05'), pd.Timestamp('2017-08-06'), pd.Timestamp('2017-08-07')]



forecast = pd.Series(results_ARIMA.forecast(steps=7)[0],dates)

forecast = np.exp(forecast)

print(forecast)

error = mean_squared_error(test, forecast)

print('Test MSE: %.3f' % error)
plt.plot(forecast, color ='green', label ='Predicted rates')

plt.plot(test, color = 'red', label = 'Observed from test data')

plt.title('RMSE: %.4f'% np.sqrt(sum((forecast-test)**2)/len(data)))

plt.legend(loc = 'best')
predictions_MA_diff = pd.Series(results_MA.fittedvalues, copy=True)

print(predictions_MA_diff.head())
predictions_MA_diff_cumsum = predictions_MA_diff.cumsum()

print(predictions_MA_diff_cumsum.head())
predictions_MA_log = pd.Series(ts_logtransformed.iloc[0], index=ts_logtransformed.index)

predictions_MA_log = predictions_MA_log.add(predictions_MA_diff_cumsum,fill_value=0)

predictions_MA_log.head()
predictions_MA = np.exp(predictions_MA_log)

plt.plot(data)

plt.plot(predictions_MA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_MA-data)**2)/len(data)))
dates = [pd.Timestamp('2017-08-01'), pd.Timestamp('2017-08-02'), pd.Timestamp('2017-08-03'),pd.Timestamp('2017-08-04'), pd.Timestamp('2017-08-05'), pd.Timestamp('2017-08-06'), pd.Timestamp('2017-08-07')]



forecast = pd.Series(results_MA.forecast(steps=7)[0],dates)

forecast = np.exp(forecast)

print(forecast)

error = mean_squared_error(test, forecast)

print('Test MSE: %.3f' % error)
plt.plot(forecast, color ='green', label ='Predicted rates')

plt.plot(test, color = 'red', label = 'Observed from test data')

plt.title('RMSE: %.4f'% np.sqrt(sum((forecast-test)**2)/len(data)))

plt.legend(loc = 'best')
monthly_mean = data.resample('M').mean()

monthly_mean

print(monthly_mean.head(13))

monthly_mean.plot()
test_logtransformed = np.log(test)
history = [x for x in ts_logtransformed]

predictions = list()

for t in range(len(test)):

    output = results_MA.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test_logtransformed[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test_logtransformed, predictions)

print('Test MSE: %.3f' % error)

from fbprophet import Prophet
data.head()
data_prophet = data.copy()

data_prophet = pd.DataFrame(data_prophet)

data_prophet.reset_index(drop=False, inplace=True)

data_prophet.columns =['ds','y']

data_prophet
m = Prophet()

m.fit(data_prophet)

future = m.make_future_dataframe(periods=7, freq='D')

forecast = m.predict(future)

m.plot(forecast)

data.plot()

m.plot_components(forecast)
forecast.columns
forecasted_values = forecast[['ds', 'yhat']].tail(7)
forecasted_values = forecasted_values.set_index('ds')

forecasted_values.columns = ['y']

forecasted_values
mean_squared_error(forecasted_values['y'],test)