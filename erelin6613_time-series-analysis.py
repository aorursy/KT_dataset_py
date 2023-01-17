!pip3 install yfinance --quiet
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torch.optim import Adam

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 6);
sns.set_style('whitegrid')
fundamentals = pd.read_csv('../input/nyse/fundamentals.csv')
prices = pd.read_csv('../input/nyse/prices.csv')
securities = pd.read_csv('../input/nyse/securities.csv')
price_split = pd.read_csv('../input/nyse/prices-split-adjusted.csv')
fundamentals.head()
prices.head()
securities.head()
price_split.head()
price_split['symbol'].unique()[10:50]
stock = 'EA'
df = price_split.loc[price_split['symbol']==stock]
df
def update_df(df, keep_split=True):
    df.index = pd.to_datetime(df['date'])
    stock = df['symbol'].values[0]
    start = str(df.index[-1]).split(' ')[0]
    end = str(datetime.datetime.now()).split(' ')[0]
    ext = yfinance.download(stock, start, end)
    ext.columns = [x.lower() for x in ext.columns]
    ext.drop('adj close', axis=1, inplace=True)
    ext['symbol'] = stock
    if keep_split:
        return df, ext
    return pd.concat([df, ext], axis=0).drop('date', axis=1)

train_set, test_set = update_df(df)
train_set
fig, ax = plt.subplots(1, 2)
ax[0].plot(df['close'])
ax[1].plot(df['close'].diff())
sample_range = pd.date_range(train_set.index[-200], periods=200, freq='D')
sample_range
sample = train_set.loc[sample_range[0]:sample_range[-1], 'close']
plt.plot(sample, label='original')
plt.plot(sample.rolling(30).mean(), label='MA 30 days', linewidth=2.5)
plt.plot(sample.rolling(7).mean(), label='MA 7 days', linewidth=2.5)
plt.legend();
train_set
lags = [x*5 for x in range(0, 70)]
plot_pacf(train_set['close'], lags=lags);
plot_acf(train_set['close'], lags=lags);
sns.distplot(train_set['close'].diff())
results = adfuller(train_set['close'].dropna())
results
plt.plot(train_set['close'].diff().dropna())
p, d, q = 2, 1, 2
model = SARIMAX(train_set['close'], order=(p, d, q), seasonal_order=(2, 1, 2, 5))
m_res = model.fit()
forecast = m_res.forecast(steps=1200)
ix = pd.date_range(train_set.index[-1], periods=len(forecast), freq='D')
forecast = pd.DataFrame({'close': forecast.values}, index=ix)
forecast
plt.plot(train_set['close'], label='historical prices'); #.predicted_mean);
plt.plot(test_set['close'], label='test prices');
plt.plot(forecast, label='predicted prices');
plt.legend();
!pip3 install fbprophet
from fbprophet import Prophet

ch_p = train_set.loc[train_set['close']==train_set['close'].max()].index
ch_p = ch_p.to_pydatetime()
print(ch_p)

prophet_model = Prophet(uncertainty_samples=100) #, changepoints=ch_p)
train_set['y'] = train_set['close']
train_set['ds'] = train_set.index
prophet_model.fit(train_set.loc[:, ['ds', 'y']])
future = prophet_model.make_future_dataframe(periods=1500)
forecast = prophet_model.predict(future)
forecast.index = pd.to_datetime(forecast['ds'])
forecast.drop('ds', axis=1)

plt.plot(train_set['close']);
plt.plot(test_set['close'], label='test prices');
plt.plot(forecast['yhat'], label='predicted prices');
plt.legend();
train_set.drop(['y', 'ds'], axis=1, inplace=True)
train_set
pd.to_datetime(test_set.index)
# help(ExponentialSmoothing)
exp_smooth = ExponentialSmoothing(train_set['close'].asfreq('D').fillna(method='ffill'), trend='mul', seasonal='mul', seasonal_periods=180)#, damped=True, seasonal='add', freq='D')
results = exp_smooth.fit(use_basinhopping=True)
forecast = results.forecast(steps=len(test_set))
forecast.index = test_set.index
forecast.plot()
test_set['close'].plot()
holt = Holt(train_set['close'].asfreq('D').fillna(method='ffill'), 
            exponential=True) #, trend='mul', seasonal='add', seasonal_periods=127)#, damped=True, seasonal='add', freq='D')
results = holt.fit()
forecast = results.forecast(steps=len(test_set))
forecast.index = test_set.index
forecast.plot()
test_set['close'].plot()