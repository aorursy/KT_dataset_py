# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

from datetime import datetime

from IPython.display import Image
dates=np.arange('2020-01-01', '2020-01-31',  dtype='datetime64[D]')

dates
dates=np.arange('2020-01-01', '2020-01-31', 10, dtype='datetime64[D]')

dates
import pandas as pd

dates=pd.date_range('2020-01-01', periods=7, freq='D')

dates
data=pd.read_csv('../input/starbucks/starbucks.csv')

data.head()
data.info()
data['Date']=pd.to_datetime(data['Date'])
data.info()
data=data.set_index('Date')

data.head()
data.index
# daily to yearly

data.resample('A').mean()
data.head()
# Let's resample only a series column and plot it



data['Close'].resample('A').mean().plot(kind='bar')
# shifting means time shifting the whole dataframe or a particular series by 1 up or down



data.head()
data.shift(1)
data.rolling(window=3).mean().head()

# Notice that first 2 values are NAN hence sampling mean of 3 samples will start only from the 3rd sample
data['Close'].plot(figsize=(12,5))

data['Close'].rolling(window=30).mean().plot()
# here we only focus on the plot containing the data from 2016-2017. we can also limit the data in the plot itself using xlim and ylim but i suggest to limit the data before plotting

data['Close']['2016-01-01':'2017-12-31'].plot(figsize=(12,5), ls='--', color='red')

# Dates are separated by a comma:

data['Close'].plot(figsize=(12,4),xlim=['2017-01-01','2017-03-01']);
df = pd.read_csv('../input/macrodata/macrodata.csv',index_col=0,parse_dates=True)

df.head()
ax = df['realgdp'].plot()

ax.autoscale(axis='x',tight=True)

ax.set(ylabel='REAL GDP');
from statsmodels.tsa.filters.hp_filter import hpfilter



# Tuple unpacking

gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb=1600)
df['trend'] = gdp_trend

df[['trend','realgdp']].plot().autoscale(axis='x',tight=True);
df[['trend','realgdp']]['2000-03-31':].plot(figsize=(12,8)).autoscale(axis='x',tight=True);
airline = pd.read_csv('../input/airline/airline_passengers.csv',index_col='Month',parse_dates=True)

airline.dropna(inplace=True)

airline.head()
airline.plot();
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')  # model='mul' also works

result.plot();
airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()

airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()

airline.head(15)
airline.plot();
airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12,adjust=False).mean()
airline[['Thousands of Passengers','EWMA12']].plot();
airline[['Thousands of Passengers','EWMA12','12-month-SMA']].plot(figsize=(12,8)).autoscale(axis='x',tight=True);
df = pd.read_csv('../input/airline/airline_passengers.csv',index_col='Month',parse_dates=True)

df.dropna(inplace=True)

df.index
df.index.freq = 'MS'

df.index
from statsmodels.tsa.holtwinters import SimpleExpSmoothing



span = 12

alpha = 2/(span+1)



df['EWMA12'] = df['Thousands of Passengers'].ewm(alpha=alpha,adjust=False).mean()

df['SES12']=SimpleExpSmoothing(df['Thousands of Passengers']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)

df.head()
from statsmodels.tsa.holtwinters import ExponentialSmoothing



df['DESadd12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend='add').fit().fittedvalues.shift(-1)

df.head()
df[['Thousands of Passengers','EWMA12','DESadd12']].iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True);
df['DESmul12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend='mul').fit().fittedvalues.shift(-1)

df.head()
df[['Thousands of Passengers','DESadd12','DESmul12']].iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True);
df = pd.read_csv('../input/airline/airline_passengers.csv',index_col='Month',parse_dates=True)

df.index.freq = 'MS'

df.head()
train_data = df.iloc[:109] # Goes up to but not including 109

test_data = df.iloc[108:]
from statsmodels.tsa.holtwinters import ExponentialSmoothing



fitted_model=ExponentialSmoothing(train_data['Thousands of Passengers'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
# Let's do the forecast. Ignore the warings as it is related to statsmodels

predictions=fitted_model.forecast(36).rename("Forecast")

predictions
train_data['Thousands of Passengers'].plot(legend=True, label="TRAIN DATA")

test_data['Thousands of Passengers'].plot(legend=True, label="TEST DATA", figsize=(12,8))

predictions.plot(legend=True, label='PREDICTIONS')
train_data['Thousands of Passengers'].plot(legend=True,label='TRAIN')

test_data['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8))

predictions.plot(legend=True,label='PREDICTION',xlim=['1958-01-01','1961-01-01'])
from sklearn.metrics import mean_squared_error,mean_absolute_error
mean_absolute_error(test_data,predictions)
np.sqrt(mean_squared_error(test_data,predictions))
df2 = pd.read_csv('../input/samples/samples.csv',index_col=0,parse_dates=True)

df2.head()
df2['a'].plot(ylim=[0,100],title="STATIONARY DATA").autoscale(axis='x',tight=True);
df2['b'].plot(ylim=[0,100],title="NON-STATIONARY DATA").autoscale(axis='x',tight=True);
from statsmodels.tsa.statespace.tools import diff

df2['d1'] = diff(df2['b'],k_diff=1)



df2['d1'].plot(title="FIRST DIFFERENCE DATA").autoscale(axis='x',tight=True);


# Load a non-stationary dataset

df1 = pd.read_csv('../input/airline/airline_passengers.csv',index_col='Month',parse_dates=True)

df1.index.freq = 'MS'



# Load a stationary dataset

df2 = pd.read_csv('../input/dailytotalfemalebirths/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)

df2.index.freq = 'D'
# Import the models

from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols
#Ignore warnings

import warnings

warnings.filterwarnings('ignore')
df = pd.DataFrame({'a':[13, 5, 11, 12, 9]})

arr = acovf(df['a'])

arr
arr2 = acovf(df['a'],unbiased=True)

arr2
arr3 = acf(df['a'])

arr3
arr4 = pacf_yw(df['a'],nlags=4,method='mle')

arr4
arr5 = pacf_yw(df['a'],nlags=4,method='unbiased')

arr5
from pandas.plotting import lag_plot



lag_plot(df1['Thousands of Passengers']);
lag_plot(df2['Births']);
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# Let's look first at the ACF array. By default acf() returns 40 lags

acf(df2['Births'])
# Now let's plot the autocorrelation at different lags

title = 'Autocorrelation: Daily Female Births'

lags = 40

plot_acf(df2,title=title,lags=lags);
acf(df1['Thousands of Passengers'])
title = 'Autocorrelation: Airline Passengers'

lags = 40

plot_acf(df1,title=title,lags=lags);
# Load a seasonal dataset

df1 = pd.read_csv('../input/airline/airline_passengers.csv',index_col='Month',parse_dates=True)

df1.index.freq = 'MS'



# Load a nonseasonal dataset

df2 = pd.read_csv('../input/dailytotalfemalebirths/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)

df2.index.freq = 'D'
df1['12-month-SMA'] = df1['Thousands of Passengers'].rolling(window=12).mean()

df1['12-month-Std'] = df1['Thousands of Passengers'].rolling(window=12).std()



df1[['Thousands of Passengers','12-month-SMA','12-month-Std']].plot();
from statsmodels.tsa.stattools import adfuller

print('Augmented Dickey-Fuller Test on Airline Data')

dftest = adfuller(df1['Thousands of Passengers'],autolag='AIC')

dftest
help(adfuller)
print('Augmented Dickey-Fuller Test on Airline Data')



dfout = pd.Series(dftest[0:4],index=['ADF test statistic','p-value','# lags used','# observations'])



for key,val in dftest[4].items():

    dfout[f'critical value ({key})']=val

print(dfout)
from statsmodels.graphics.tsaplots import month_plot,quarter_plot



# Note: add a semicolon to prevent two plots being displayed in jupyter

month_plot(df1['Thousands of Passengers']);
dfq = df1['Thousands of Passengers'].resample(rule='Q').mean()



quarter_plot(dfq);
# Load a non-stationary dataset

df1 = pd.read_csv('../input/airline/airline_passengers.csv',index_col='Month',parse_dates=True)

df1.index.freq = 'MS'



# Load a stationary dataset

df2 = pd.read_csv('../input/dailytotalfemalebirths/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)

df2.index.freq = 'D'
!pip install pmdarima
from pmdarima import auto_arima



# Ignore harmless warnings

import warnings

warnings.filterwarnings("ignore")
help(auto_arima)
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(df2['Births']).plot();
auto_arima(df2['Births'],error_action='ignore').summary()
arima_model=auto_arima(df2['Births'], start_p=0, start_q=0, d=None, seasonal=False, max_p=5, max_q=5, error_action='ignore', stepwise=True, suppress_warnings=True)
arima_model.summary()
# Load specific forecasting tools

from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders

from pmdarima import auto_arima # for determining ARIMA orders





df2 = pd.read_csv('../input/tradeinvestories/TradeInventories.csv',index_col='Date',parse_dates=True)

df2.index.freq='MS'
title = 'Real Manufacturing and Trade Inventories'

ylabel='Chained 2012 Dollars'

xlabel='' # we don't really need a label here



ax = df2['Inventories'].plot(figsize=(12,5),title=title)

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel)

result=seasonal_decompose(df2['Inventories'], model='additive').plot()
auto_arima(df2['Inventories'], seasonal=False).summary()
(df2['Inventories']-df2['Inventories'].shift(1)).plot()
model_arima_1 = auto_arima(df2['Inventories'], start_p=0, start_q=0,

                          max_p=2, max_q=2, m=12,

                          seasonal=False,

                          d=None, trace=True,

                          error_action='ignore',   # we don't want to know if an order does not work

                          suppress_warnings=True,  # we don't want convergence warnings

                          stepwise=True)           # set to stepwise



model_arima_1.summary()
len(df2)
# Set one year for testing

train = df2.iloc[:252]

test = df2.iloc[252:]
# Fit the ARIMA model . Let's use 1,1,1 order as auto arima selected 0,1,0 due to early stopping but it seems 1,1,1 is much better

model = ARIMA(train['Inventories'],order=(1,1,1))

results = model.fit()

results.summary()
# Obtain predicted values

start=len(train)

end=len(train)+len(test)-1

predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(1,1,1) Predictions')
ax = test['Inventories'].plot(legend=True,figsize=(12,6),title=title)

predictions.plot(legend=True)

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel)

model = ARIMA(df2['Inventories'],order=(1,1,1))

results = model.fit()

fcast = results.predict(len(df2),len(df2)+11,typ='levels').rename('ARIMA(1,1,1) Forecast')
ax = df2['Inventories'].plot(legend=True,figsize=(12,6),title=title)

fcast.plot(legend=True)

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel)
from statsmodels.tsa.statespace.sarimax import SARIMAX
df=pd.read_csv('../input/co2-mm/co2_mm_mlo.csv')

df.head()
# let's create a datetime column

df['Date']=pd.to_datetime({'year': df['year'], 'month':df['month'], 'day':1})
df=df.set_index('Date')
df.info()
seasonal_check=seasonal_decompose(df['interpolated'], model='add').plot()
auto_arima(df['interpolated'], seasonal=True, m=12).summary()
len(df)
train=df.iloc[:717]

test=df.iloc[717:]
model=SARIMAX(train['interpolated'], order=(2,1,1), seasonal_order=(1,0,1,12) )
results=model.fit()
results.summary()
start=len(train)

end=len(train)+len(test)-1
predictions=results.predict(start, end, typ='levels').rename("SARIMA predictions")
test['interpolated'].plot(legend=True, figsize=(12,8))

predictions.plot(legend=True)