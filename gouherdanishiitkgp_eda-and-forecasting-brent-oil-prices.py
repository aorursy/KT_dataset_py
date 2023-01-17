import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/brent-oil-prices"))
df = pd.read_csv("../input/brent-oil-prices/BrentOilPrices.csv")
df.head()
import seaborn as sns

from matplotlib import pyplot as plt



df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y")

df.head()
g = sns.lineplot(x='Date',y='Price',data = df)

plt.title("Brent Oil Price Trend")
def plot_price_trend(df, start_date, end_date):

    """

    This function filters the dataframe for the specified date range and 

    plots the line plot of the data using seaborn.

    

    The dataframe may not be indexed on any Datetime column.

    In this case, we use mask to filter out the date.

    

    PS - There is another function provided later in the notebook 

    which used indexed column to filter data

    """

    mask = (df['Date'] > start_date) & (df['Date'] <= end_date)

    sdf = df.loc[mask]

    plt.figure(figsize = (10,5))

    chart = sns.lineplot(x='Date',y='Price',data = sdf)

#     chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

    plt.title("Brent Oil Price Trend")
plot_price_trend(df,'2017-01-01','2019-01-01')
from fbprophet import Prophet

m = Prophet()
pro_df = df

pro_df.columns = ['ds','y']

pro_df.head()
m.fit(pro_df)

future = m.make_future_dataframe(periods = 90)

forecast = m.predict(future)
forecast.head()
m.plot_components(forecast)
m.plot(forecast)
cmp_df = forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']].join(pro_df.set_index('ds'))
cmp_df.head()
cmp_df.tail(5)
plt.figure(figsize=(17,8))

#plt.plot(cmp_df['yhat_lower'])

#plt.plot(cmp_df['yhat_upper'])

plt.plot(cmp_df['yhat'])

plt.plot(cmp_df['y'])

plt.legend()

plt.show()
def plot_price_forecast(df,start_date, end_date):

    """

    This function filters the dataframe for the specified date range and 

    plots the actual and forecast data.

    

    Assumption: 

    - The dataframe has to be indexed on a Datetime column

    This makes the filtering very easy in pandas using df.loc

    """

    cmp_df = df.loc[start_date:end_date]

    plt.figure(figsize=(17,8))

    plt.plot(cmp_df['yhat'])

    plt.plot(cmp_df['y'])

    plt.legend()

    plt.show()
plot_price_forecast(cmp_df,'2017-01-01','2020-01-01')
from statsmodels.tsa.arima_model import ARIMA    # ARIMA Modeling

from statsmodels.tsa.stattools import adfuller   # Augmented Dickey-Fuller Test for Checking Stationary

from statsmodels.tsa.stattools import acf, pacf  # Finding ARIMA parameters using Autocorrelation

from statsmodels.tsa.seasonal import seasonal_decompose # Decompose the ARIMA Forecast model
arima_df = df.set_index('ds')

arima_df.head()
# Perform Augmented Dickey–Fuller test to check if the given Time series is stationary:

def test_stationarity(ts):

    

    #Determing rolling statistics

    rolmean = ts.rolling(window=12).mean()

    rolstd = ts.rolling(window=12).std()



    #Plot rolling statistics:

    orig = plt.plot(ts, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(ts['y'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
test_stationarity(arima_df)
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_acf(arima_df)

plot_pacf(arima_df)
# Implementing own function to create ACF plot

def get_acf_plot(ts):

    #calling acf function from stattools

    y = ts['y']

    lag_acf = acf(y, nlags=500)

    plt.figure(figsize=(16, 7))

    plt.plot(lag_acf, marker="o")

    plt.axhline(y=0,linestyle='--',color='gray')

    plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')

    plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')

    plt.title('Autocorrelation Function')

    plt.xlabel('number of lags')

    plt.ylabel('correlation')

    

def get_pacf_plot(ts):

    #calling pacf function from stattools

    y = arima_df['y']

    lag_pacf = pacf(y, nlags=50)

    plt.figure(figsize=(16, 7))

    plt.plot(lag_pacf, marker="o")

    plt.axhline(y=0,linestyle='--',color='gray')

    plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')

    plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')

    plt.title('Partial Autocorrelation Function')

    plt.xlabel('number of lags')

    plt.ylabel('correlation')
get_acf_plot(arima_df)

get_pacf_plot(arima_df)
# Log Transformation

ts_log = np.log(arima_df)

plt.plot(ts_log)
# Moving Average of last 12 values

moving_avg = ts_log.rolling(12).mean()

plt.plot(ts_log)

plt.plot(moving_avg, color='red')
# Differencing

ts_log_ma_diff = ts_log - moving_avg

ts_log_ma_diff.head(12)
ts_log_ma_diff.dropna(inplace=True)

test_stationarity(ts_log_ma_diff)
# Exponentially weighted moving average 

expwighted_avg = ts_log.ewm(halflife=12).mean()



plt.plot(ts_log)

plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = ts_log - expwighted_avg

test_stationarity(ts_log_ewma_diff)
ts_log_diff = ts_log - ts_log.shift()

plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)

test_stationarity(ts_log_diff)
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_log, freq = 30)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(ts_log, label='Original')

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
ts_log_decompose = residual

ts_log_decompose.dropna(inplace=True)

test_stationarity(ts_log_decompose)
model = ARIMA(ts_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(ts_log_diff)

plt.plot(results_ARIMA.fittedvalues, color='red')

# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))