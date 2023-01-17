!pip install pmdarima
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import acf, pacf

import warnings

warnings.filterwarnings("ignore")
data_path = r'/kaggle/input/airpassengers/AirPassengers.csv'
data = pd.read_csv(data_path)

print('Data sample \n', data.head())

print('Shape of Data is ', data.shape)
data.info()
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

data = pd.read_csv(data_path, parse_dates=['Month'], index_col='Month',date_parser=dateparse)

data.head()
data.info()
%matplotlib inline
plt.plot(data)
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data, model='multiplicative')

result.plot()

plt.show()
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=12).mean()

    rolstd = timeseries.rolling(window=12).std()



    #Plot rolling statistics:

    plt.figure(figsize=(12,5), dpi=100)

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Augmented Dickey-Fuller (ADF) test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

    

    #KPSS checks for trend stationarity

    print('\nResults of KPSS Test:')

    kpsstest = kpss(timeseries, regression='c')

    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])

    for key,value in kpsstest[3].items():

        kpss_output['Critical Value (%s)'%key] = value

    print (kpss_output)

    
test_stationarity(data['#Passengers'])
Dp = data['#Passengers'] - data['#Passengers'].shift(1)
Dp.dropna(inplace = True)
plt.figure(figsize=(12,5), dpi=100)

Dp.plot()
plt.figure(figsize=(12,5), dpi=100)

plt.plot(Dp - Dp.shift(12))  #difference seasonal components
Dp = np.log(data['#Passengers'])
plt.figure(figsize=(12,5), dpi=100)

plt.plot(Dp)  #variance stabilized
Dp = Dp - Dp.shift(1) #stabilize mean
plt.plot(Dp)  #mean stabilized and variance stabilized
%matplotlib inline

test_stationarity(Dp.dropna())
from scipy.stats import boxcox

from scipy.special import inv_boxcox
series, lam = boxcox(data['#Passengers'])  #stabilize variance

Dp = series - pd.Series(series).shift(1)   #stabilize mean
%matplotlib inline



fix, ax = plt.subplots(3, figsize=(12,10))

ax[0].plot(series)

ax[1].plot(Dp)  #plot differenced series

ax[2].hist(Dp)  #histogram of differenced series

plt.show()

print('lambda', lam)
%matplotlib inline

test_stationarity(Dp.dropna())
from sklearn.linear_model import LinearRegression

y = data['#Passengers'].to_numpy()

x = np.arange(0, data.shape[0])

x = x.reshape(-1, 1)

model = LinearRegression()

model.fit(x, y)



trend = model.predict(x)

plt.figure(figsize=(12,5), dpi=100)

plt.plot(y)

plt.plot(trend)

plt.show()



detrended = y - trend

test_stationarity(pd.Series(detrended))

%matplotlib inline

from sklearn.linear_model import LinearRegression

y = np.log(data['#Passengers'].to_numpy())

x = np.arange(0, data.shape[0])

x = x.reshape(-1, 1)

model = LinearRegression()

model.fit(x, y)



trend = model.predict(x)

plt.figure(figsize=(12,5), dpi=100)

plt.plot(y)

plt.plot(trend)

plt.show()



detrended = y - trend

test_stationarity(pd.Series(detrended))



# Accuracy metrics

def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    mins = np.amin(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    maxs = np.amax(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    minmax = 1 - np.mean(mins/maxs)             # minmax

    acf1 = acf(forecast - actual)[1]                      # ACF1

    return({'mape':mape, 'me':me, 'mae': mae, 

            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 

            'corr':corr, 'minmax':minmax})
%matplotlib inline

# Original Series



fig, axes = plt.subplots(3, 2, figsize = (15,10))



axes[0, 0].plot(series)

axes[0, 0].set_title('Original Series')

x  = plot_pacf(series, ax=axes[0, 1])



# 1st Differencing

axes[1, 0].plot(pd.Series(series).diff())

axes[1, 0].set_title('1st Order Differencing')

x = plot_pacf(pd.Series(series).diff().dropna(), ax=axes[1, 1])



# 2nd Differencing

axes[2, 0].plot(pd.Series(series).diff().diff())

axes[2, 0].set_title('2nd Order Differencing')

x = plot_pacf(pd.Series(series).diff().dropna(), ax=axes[2, 1])
from statsmodels.tsa.arima_model import ARIMA
#Dp = np.log(data['#Passengers'])

model = ARIMA(series, order=(2, 1, 0))  

results_AR = model.fit()  

plt.figure(figsize=(12,5), dpi=100)

plt.plot(Dp)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-Dp.dropna())**2))
len(results_AR.resid)
print(results_AR.summary())
# Original Series

fig, axes = plt.subplots(3, 2, figsize=(15,10))

axes[0, 0].plot(series); axes[0, 0].set_title('Original Series')

x = plot_acf(series, ax=axes[0, 1])



# 1st Differencing

axes[1, 0].plot(pd.Series(series).diff()); axes[1, 0].set_title('1st Order Differencing')

x = plot_acf(pd.Series(series).diff().dropna(), ax=axes[1, 1])



# 2nd Differencing

axes[2, 0].plot(pd.Series(series).diff().diff()); axes[2, 0].set_title('2nd Order Differencing')

x = plot_acf(pd.Series(series).diff().dropna(), ax=axes[2, 1])

model = ARIMA(series, order=(0, 1, 2))  

results_MA = model.fit(disp=-1)  

plt.figure(figsize=(12,5), dpi=100)

plt.plot(Dp)

plt.plot(results_MA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_MA.fittedvalues - Dp.dropna())**2))
print(results_MA.summary())
plt.figure(figsize=(12,5), dpi=100)

model = ARIMA(series, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(Dp.dropna())

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - Dp.dropna())**2))
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

predictions_ARIMA_diff
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_diff_cumsum
predictions_ARIMA_log = pd.Series(series[0], index  = np.arange(0,144))

predictions_ARIMA_log
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA_log
predictions_ARIMA_log.index  = data['#Passengers'].index

predictions_ARIMA_log.index
plt.figure(figsize=(12,5), dpi=100)

predictions_ARIMA = inv_boxcox(predictions_ARIMA_log, lam)

plt.plot(data['#Passengers'])

plt.plot(predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA - data['#Passengers'])**2)/len(data['#Passengers'])))
def rescale(fittedvalues, lam, data):

    predictions_ARIMA_diff = pd.Series(fittedvalues, copy=True)

    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

    predictions_ARIMA_log = pd.Series(series[0], index  = np.arange(0,len(fittedvalues) + 1))

    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

    predictions_ARIMA_log.index  = data['#Passengers'].index

    predictions_ARIMA = inv_boxcox(predictions_ARIMA_log, lam)

    plt.plot(data['#Passengers'])

    plt.plot(predictions_ARIMA)

    plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA - data['#Passengers'])**2)/len(data['#Passengers'])))

    

    return predictions_ARIMA
plt.figure(figsize=(12,5), dpi=100)

predictions_ARIMA = rescale(results_ARIMA.fittedvalues, lam, data)
forecast_accuracy(predictions_ARIMA, data['#Passengers'])
from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm
series, lam = boxcox(data['#Passengers'])

model = pm.auto_arima(series, start_p=0, start_q=0,

                      test='adf',       # use adftest to find optimal 'd'

                      max_p=5, max_q=5, # maximum p and q

                      m=1,              # frequency of series

                      d=1,           # let model determine 'd'

                      seasonal=False,   # No Seasonality

                      start_P=0, 

                      D=0, 

                      trace=True,

                      error_action='ignore',  

                      suppress_warnings=True, 

                      stepwise=True)



print(model.summary())
series, lam = boxcox(data['#Passengers'])

Training_Ratio = 0.8

Num_Train_Points = int(Training_Ratio * series.shape[0])

train_ts = series[0:Num_Train_Points]

test_ts =  series[Num_Train_Points : ]
model = pm.auto_arima(train_ts, start_p=0, start_q=0,

                      test='adf',       # use adftest to find optimal 'd'

                      max_p=5, max_q=5, # maximum p and q

                      m=1,              # frequency of series

                      d=1,           # let model determine 'd'

                      seasonal=False,   # No Seasonality

                      start_P=0, 

                      D=0, 

                      trace=True,

                      error_action='ignore',  

                      suppress_warnings=True, 

                      stepwise=True)



print(model.summary())
model_final = ARIMA(train_ts, order=(2, 1, 1))  

results_ARIMA = model_final.fit(disp=-1)  

Dp = train_ts - pd.Series(train_ts).shift(1)   #stabilize mean



plt.figure(figsize=(12,5), dpi=100)

plt.plot(Dp.dropna())

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - Dp.dropna())**2))

print(results_ARIMA.summary())
plt.figure(figsize=(12,5), dpi=100)

predictions_ARIMA = rescale(results_ARIMA.fittedvalues, lam, data[0:len(train_ts)])
forecast_accuracy(predictions_ARIMA, data.ix[0:len(train_ts), '#Passengers'])
fc, se, conf = results_ARIMA.forecast(29, alpha=0.05)  # 95% conf
fc = inv_boxcox(fc, lam)

se = inv_boxcox(se, lam)

conf = inv_boxcox(conf, lam)
test_ts
# Make as pandas series

fc_series = pd.Series(fc, index=data.index[Num_Train_Points:])

lower_series = pd.Series(conf[:, 0], index=data.index[Num_Train_Points:])

upper_series = pd.Series(conf[:, 1], index=data.index[Num_Train_Points:])
train_series = pd.Series(inv_boxcox(train_ts, lam), index=data.index[0: Num_Train_Points])

test_series = pd.Series(inv_boxcox(test_ts, lam), index=data.index[Num_Train_Points: ])
# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train_series, label='training')

plt.plot(test_series, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
smodel = pm.auto_arima(train_ts, start_p=0, start_q=0,

                         test='adf',

                         max_p=3, max_q=3, m=12,

                         start_P=0, seasonal=True,

                         d=None, D=1, trace=True,

                         error_action='ignore',  

                         suppress_warnings=True, 

                         stepwise=True)



smodel.summary()
# Forecast

n_periods = 29

fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
# make series for plotting purpose

fitted_series = pd.Series(inv_boxcox(fitted, lam), index=data.index[Num_Train_Points:])

lower_series = pd.Series(inv_boxcox(confint[:, 0], lam), index=data.index[Num_Train_Points:])

upper_series = pd.Series(inv_boxcox(confint[:, 1], lam), index=data.index[Num_Train_Points:])



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train_series, label='training')

plt.plot(test_series, label='actual')

plt.plot(fitted_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

plt.legend(loc='upper left', fontsize=8)



plt.title("SARIMA - Forecast vs Actuals")

plt.show()



print('Accuracy Metrics:')

forecast_accuracy(fitted_series, data.ix[len(train_ts):, '#Passengers'])
fitted_values_train = pd.Series(inv_boxcox(smodel.predict_in_sample(start = 0, end = len(train_ts) - 1), lam), index = train_series.index)

                           

plt.figure(figsize=(12,5), dpi=100)

plt.plot(fitted_values_train, label = 'fitted values')

plt.plot(train_series, label = 'actual training')



plt.title('Fitted vs Actuals - Training Period')

plt.legend(loc='upper left', fontsize=8)

plt.show()



forecast_accuracy(fitted_values_train, data.ix[0:len(train_ts), '#Passengers'])
x = smodel.plot_diagnostics(figsize=(15,10))