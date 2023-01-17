import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from tqdm import tqdm
plt.style.use('ggplot')
INPUT_DIR = '../input/m5-forecasting-accuracy'
cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')

stv.head()
#select all the columns that contain d_ (sales data columns)
d_cols = [c for c in stv.columns if 'd_' in c] 
# Unit sales of all products, aggregated for all stores/states
all_data = stv[d_cols] \
    .sum(axis=0) \
    .T \
    .reset_index()

all_data.columns = ['d','sales']

print(all_data)
cal.head()
# Checking events
print(cal['event_name_1'].unique())
print(cal['event_name_2'].unique())
# Merge calendar on our items' data
all_data_merged = all_data.merge(cal, how='left', validate='1:1')
all_data_merged['date'] = pd.to_datetime(all_data_merged['date'])
all_data_merged.head()
#final time series
ts = all_data_merged.set_index('date')['sales']

#Detect days that have either event_1 or event_2
places = all_data_merged.loc[~(all_data_merged['event_name_1'].isna()) | ~(all_data_merged['event_name_2'].isna())]['d']

change = list(all_data_merged.d.isin(list(places)))

for i in range(len(change)):
    if change[i] == True:
        ts.iloc[i] = (ts.iloc[i-1] + ts.iloc[i+1]) / 2

#visualise the time series
ts.plot(figsize=(20, 5),
          title='unit sales of all products by day',color = 'blue')
plt.legend('')
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.figure(figsize=(20,10) )
plt.subplot(411)
plt.plot(ts,label = 'Original',color = 'blue')
plt.legend(loc='upper right')
plt.subplot(412)
plt.plot(trend,label = 'Trend',color = 'blue')
plt.legend(loc='upper right')
plt.subplot(413)
plt.plot(seasonal,label = 'Seasonality',color = 'blue')
plt.legend(loc='upper right')
plt.subplot(414)
plt.plot(residual,label = 'Residual',color = 'blue')
plt.legend(loc='upper right')
plt.tight_layout()
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.figure(figsize=(20,10))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='upper right')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(ts)
def sarima_train_test(t_series, p = 2, d = 1, r = 2, NUM_TO_FORECAST = 56, do_plot_results = True):
    NUM_TO_FORECAST = NUM_TO_FORECAST  # Similar to train test splits.
    dates = np.arange(t_series.shape[0])

    model = SARIMAX(t_series, order = (p, d, r), trend = 'c')
    results = model.fit()
    results.plot_diagnostics(figsize=(18, 14))
    plt.show()

    forecast = results.get_prediction(start = -NUM_TO_FORECAST)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Plot the forecast
    plt.figure(figsize=(14,16))
    plt.plot(dates[-NUM_TO_FORECAST:],
            mean_forecast.values,
            color = 'red',
            label = 'forecast')


    plt.plot(dates[-NUM_TO_FORECAST:],
            t_series.iloc[-NUM_TO_FORECAST:],
            color = 'blue',
            label = 'actual')
    plt.legend()
    plt.title('Predicted vs. Actual Values')
    plt.show()
    
    residuals = results.resid
    mae_sarima = np.mean(np.abs(residuals))
    print('Mean absolute error: ', mae_sarima)
    print(results.summary())
sarima_train_test(ts)
# Fine tuning was performed offline due to the kaggles run restrictions and the file was submitted