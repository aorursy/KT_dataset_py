import pandas as pd
import numpy as np
%matplotlib inline 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
#import mpld3
#mpld3.enable_notebook() #for zooming in and out of the plots
rcParams['figure.figsize']= 12,6
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
orig_dataset = pd.read_csv('../input/Prices.csv')
orig_dataset.head(), orig_dataset.tail()
# we will just be using the period until 2017, so 10 years of data

dataset = orig_dataset[:7306]
# Checking of datatypes per column
dataset.dtypes
# getting a hang of the dataset
dataset.describe()
# check for missing values
nulls_spy = dataset['SPY'].isnull()
np.sum(dataset[nulls_spy])
nulls_ticker = dataset['Ticker'].isnull()
np.sum(dataset[nulls_ticker])
dataset['Ticker'] = pd.to_datetime(dataset.Ticker)
# If there are zero values, we could see in least value since it's a float64 series
dataset['SPY'].min(), dataset['SPY'].max()
# check the range of data
dataset['Ticker'].min(), dataset['Ticker'].max()
dataset.head(), dataset.tail()
nulls_ticker = dataset['Ticker'].isnull()
np.sum(dataset[nulls_ticker])
nulls_spy = dataset['SPY'].isnull()
np.sum(dataset[nulls_spy])
# set your dates as the index for each SPY value
indexed_dataset = dataset.set_index(['Ticker'])
# just to be sure
nulls_ticker = dataset['Ticker'].isnull()
np.sum(dataset[nulls_ticker])
# plot the dataset

plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(indexed_dataset)
dftest = adfuller(indexed_dataset['SPY'], autolag ='AIC')

df_output = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', 'Lags', 'No. of observations'])

for key, value in dftest[4].items():
    df_output['Critical value (%s)'%key] = value
    
df_output
# save rolling means and std, and moving averages for original values
moving_average = indexed_dataset.rolling(window = 365).mean()
moving_std = indexed_dataset.rolling(window = 365).std()
rolmean = indexed_dataset.rolling(window = 365).mean()
rolstd = indexed_dataset.rolling(window = 365).std()

indexed_dataset_logscale = np.log(indexed_dataset)
# save moving average values in logscale

moving_average_logscale = indexed_dataset_logscale.rolling(window = 365).mean()
moving_std_logscale = indexed_dataset_logscale.rolling(window = 365).std()
# let's save a function for adf test
# the red line in  the graphs will be the moving averages with windows of a year.

def adf_test(timeseries):
    moving_average = timeseries.rolling(window=365).mean()
    moving_std = timeseries.rolling(window=365).std()
    
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(moving_average, color='red', label='Rolling Mean')
    std = plt.plot(moving_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling mean & Standard deviation')
    plt.show(block=False)    
    
    print('Results of Dickey Fuller Test:')
   
    adftest = adfuller(timeseries['SPY'], autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    print(adfoutput)
adf_test(indexed_dataset_logscale)
# now we use logscale - moving average. 
dataset_logscale_minus_moving_average = indexed_dataset_logscale - moving_average
dataset_logscale_minus_moving_average.dropna(inplace=True)
adf_test(dataset_logscale_minus_moving_average)
exponential_decay_weighted_average = indexed_dataset_logscale.ewm(halflife=365, min_periods=0, adjust=True).mean()
plt.plot(indexed_dataset_logscale)
plt.plot(exponential_decay_weighted_average, color='red')
# to cover all bases, so I also compute the differences
dataset_logscale_minus_exponential_moving_average = indexed_dataset_logscale - exponential_decay_weighted_average
adf_test(dataset_logscale_minus_exponential_moving_average)
dataset_log_diff_shifting = indexed_dataset_logscale - indexed_dataset_logscale.shift()
plt.plot(dataset_log_diff_shifting)
dataset_log_diff_shifting.dropna(inplace=True)
adf_test(dataset_log_diff_shifting)
import statsmodels.api as sm

decomposition = sm.tsa.seasonal_decompose(indexed_dataset_logscale, model='additive')
fig = decomposition.plot()
plt.show()
residual = decomposition.resid
decomposed_log_data = residual
decomposed_log_data.dropna(inplace=True)
adf_test(decomposed_log_data)
lag_acf = acf(dataset_log_diff_shifting.values, nlags=35)
lag_pacf = pacf(dataset_log_diff_shifting.values, nlags = 35, method='ols')

plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataset_log_diff_shifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataset_log_diff_shifting)), linestyle='--', color='gray')

plt.plot(lag_acf)
plt.title('Autocorrelation Function')            

import warnings   
warnings.filterwarnings("ignore")
# warning sign shows you the unsupported functionality of mpld3
#Plot PACF
plt.plot(lag_pacf)

plt.axhline(y=abs(0), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataset_log_diff_shifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataset_log_diff_shifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
# AR model
model = ARIMA(indexed_dataset_logscale, order=(1,1,1))
results_AR = model.fit(disp=-1)
plt.plot(dataset_log_diff_shifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - dataset_log_diff_shifting['SPY'])**2))
print('Plotting AR model')
# MA model
model = ARIMA(indexed_dataset_logscale, order=(1,1,1))
results_MA = model.fit(disp=-1)
plt.plot(dataset_log_diff_shifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - dataset_log_diff_shifting['SPY'])**2))
print('Plotting MA model')
# AR+I+MA = ARIMA model
model = ARIMA(indexed_dataset_logscale, order=(1,1,1))
results_ARIMA = model.fit(disp=-1)
plt.plot(dataset_log_diff_shifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - dataset_log_diff_shifting['SPY'])**2))
print('Plotting ARIMA model')
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(indexed_dataset_logscale['SPY'].iloc[0], index=indexed_dataset_logscale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()
# Inverse of log is exp. We revert back to scaling so that we could have similar values in the original dataset
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexed_dataset)
plt.plot(predictions_ARIMA)
# We provide a forecast with 95% confidence interval and the possible values as forecast.
results_ARIMA.plot_predict(1,8000)
# Now the date tomorrow is 09/09/2018
# Days from the last dataset is exactly 282 days

future = results_ARIMA.forecast(steps=282)[0]
np.exp(future[0])
from sklearn.metrics import mean_squared_error

predictions = predictions_ARIMA
expected = dataset.SPY

'''for t in range(len(expected)):
    print('predicted=%f, expected=%f' % (predictions[t], expected[t]))
    error = mean_squared_error(expected, predictions)
print('Test MSE: %.3f' % error)'''

