import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2)

import warnings 
warnings.filterwarnings('ignore')
import os
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
print("In total: ", market_train_df.shape)
market_train_df.head()
print("In total: ", news_train_df.shape)
news_train_df.head()
days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
print("In total: ", market_obs_df.shape)
market_obs_df.head()
print("In total: ", news_obs_df.shape)
news_obs_df.head()
predictions_template_df.head()
print("In market_train_df: ", market_train_df.shape);print("In market_obs_df: ", market_obs_df.shape);
print("In news_train_df: ", news_train_df.shape);print("In news_obs_df: ", news_obs_df.shape)
percent = (100 * market_train_df.isnull().sum() / market_train_df.shape[0]).sort_values(ascending=False)
percent.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by market_obs_df", fontsize = 20)
percent1 = (100 * market_obs_df.isnull().sum() / market_obs_df.shape[0]).sort_values(ascending=False)
percent1.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by market_obs_df", fontsize = 20)
news_train_df['headlineTag'].unique()[0:5]
# '' convert to NA
for i in news_train_df.columns.values.tolist():
    # Does NaN means no numbers, can '' be replaced with nan? I do not know this part.
    news_train_df[i] = news_train_df[i].replace('', np.nan)  
news_train_df['headlineTag'].unique()[0:5]
# I think it would be faster if you just replace object and categorical variables(not int,float). How do I fix the code?
percent = (100 * news_train_df.isnull().sum() / news_train_df.shape[0]).sort_values(ascending=False)
percent.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by news_train_df", fontsize = 20)
# '' convert to NA
for i in news_obs_df.columns.values.tolist():
    # Does NaN means no numbers, can '' be replaced with nan? I do not know this part.
    news_obs_df[i] = news_obs_df[i].replace('', np.nan)
percent1 = (100 * news_obs_df.isnull().sum() / news_obs_df.shape[0]).sort_values(ascending=False)
percent1.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by news_obs_df", fontsize = 20)
percent2 = (market_train_df.nunique()).sort_values(ascending=False)
percent2.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Unique Number", fontsize = 20)
plt.title("Unique Number by market_train_df", fontsize = 20)
market_train_df.nunique()
percent2 = (market_obs_df.nunique()).sort_values(ascending=False)
percent2.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Unique Number", fontsize = 20)
plt.title("Unique Number by market_obs_df", fontsize = 20)
market_obs_df.nunique()
percent2 = (news_train_df.nunique()).sort_values(ascending=False)
percent2.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Unique Number", fontsize = 20)
plt.title("Unique Number by news_train_df", fontsize = 20)
news_train_df.nunique()
percent2 = (news_obs_df.nunique()).sort_values(ascending=False)
percent2.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Unique Number", fontsize = 20)
plt.title("Unique Number by news_obs_df", fontsize = 20)
news_obs_df.nunique()
features_object = [col for col in market_train_df.columns if market_train_df[col].dtype == 'object']
features_object
market_train_df['assetCode'].value_counts()
features_object = [col for col in news_train_df.columns if news_train_df[col].dtype == 'object']
features_object
news_train_df['sourceId'].value_counts()
news_train_df['headline'].value_counts()
news_train_df['headlineTag'].value_counts()
news_train_df['provider'].value_counts()
news_train_df['subjects'].value_counts()
news_train_df['audiences'].value_counts()
(market_train_df['universe']).describe()
market_train_df['universe'].plot.hist(title = 'universe Histogram');
plt.xlabel('universe');
from sklearn.feature_extraction.text import CountVectorizer

news_train_df.head()
list(news_train_df['headline'])[0:5]
# CountVectorizer() env
news_train_df['headline'] = news_train_df['headline'].replace(np.nan, '')
news_train_df['headlineTag'] = news_train_df['headlineTag'].replace(np.nan, '')
vect = CountVectorizer()
vect.fit(list(news_train_df['headline']))
list((vect.vocabulary_).items())[0:10]
vect.vocabulary_ = sorted(vect.vocabulary_.items(), key=lambda x: x[1], reverse=True)
(vect.vocabulary_)[0:10]
vect1 = CountVectorizer(ngram_range=(2, 2))
vect1.fit(list(news_train_df['headline']))
list((vect1.vocabulary_).items())[0:10]
vect1.vocabulary_ = sorted(vect1.vocabulary_.items(), key=lambda x: x[1], reverse=True)
(vect1.vocabulary_)[0:10]
market_train_df.time.head()
market_train_df.time.tail()
market_obs_df.time.head()
market_obs_df.time.tail()
def change_date_to_datetime(x):
    str_time = str(x)
    date = '{}-{}-{}'.format(str_time[:4], str_time[5:7], str_time[8:10])
    return date

market_train_df['date'] = market_train_df['time'].apply(change_date_to_datetime)
def add_time_feature(data):
    data['date'] = pd.to_datetime(data['date'])
    data['Year'] = data.date.dt.year
    data['Month'] = data.date.dt.month
    data['Day'] = data.date.dt.day
    data['WeekOfYear'] = data.date.dt.weekofyear
    return data

market_train_df = add_time_feature(market_train_df)
best_asset_open = market_train_df.groupby("assetCode")["open"].count().to_frame().sort_values(by=['open'],ascending= False)
best_asset_open = best_asset_open.sort_values(by=['open'])
largest_by_open = list(best_asset_open.nlargest(10, ['open']).index)

best_asset_close = market_train_df.groupby("assetCode")["close"].count().to_frame().sort_values(by=['close'],ascending= False)
best_asset_close = best_asset_close.sort_values(by=['close'])
largest_by_close = list(best_asset_close.nlargest(10, ['close']).index)

best_asset_volume = market_train_df.groupby("assetCode")["volume"].count().to_frame().sort_values(by=['volume'],ascending= False)
best_asset_volume = best_asset_volume.sort_values(by=['volume'])
largest_by_volume = list(best_asset_volume.nlargest(10, ['volume']).index)
print(largest_by_open)
print(largest_by_close)
print(largest_by_volume)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
asset1Code = 'CAH.N'
asset1_df = market_train_df[(market_train_df['assetCode'] == asset1Code) & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]

asset1_df['high'] = asset1_df['open']
asset1_df['low'] = asset1_df['close']

for ind, row in asset1_df.iterrows():
    if row['close'] > row['open']:
        asset1_df.loc[ind, 'high'] = row['close']
        asset1_df.loc[ind, 'low'] = row['open']

trace1 = go.Candlestick(
    x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    open = asset1_df['open'].values,
    low = asset1_df['low'].values,
    high = asset1_df['high'].values,
    close = asset1_df['close'].values
)

layout = dict(title = "Candlestick chart for {}".format(asset1Code),
              xaxis = dict(
                  title = 'Month',
                  rangeslider = dict(visible = False)
              ),
              yaxis = dict(title = 'Price (USD)')
             )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
asset1_df = market_train_df[(market_train_df['assetCode'] == 'CAH.N') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['open'].values
    )

layout = dict(title = "Open prices of CAH.N",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
asset1_df = market_train_df[(market_train_df['assetCode'] == 'CAH.N') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values
    )

layout = dict(title = "Closing prices of CAH.N",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
asset1_df = market_train_df[(market_train_df['assetCode'] == 'CAH.N') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['volume'].values
    )

layout = dict(title = "Volume of CAH.N",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
asset1Code = 'AAPL.O'
asset1_df = market_train_df[(market_train_df['assetCode'] == asset1Code) & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]

asset1_df['high'] = asset1_df['open']
asset1_df['low'] = asset1_df['close']

for ind, row in asset1_df.iterrows():
    if row['close'] > row['open']:
        asset1_df.loc[ind, 'high'] = row['close']
        asset1_df.loc[ind, 'low'] = row['open']

trace1 = go.Candlestick(
    x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    open = asset1_df['open'].values,
    low = asset1_df['low'].values,
    high = asset1_df['high'].values,
    close = asset1_df['close'].values
)

layout = dict(title = "Candlestick chart for {}".format(asset1Code),
              xaxis = dict(
                  title = 'Month',
                  rangeslider = dict(visible = False)
              ),
              yaxis = dict(title = 'Price (USD)')
             )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
asset1_df = market_train_df[(market_train_df['assetCode'] == 'AAPL.O') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['open'].values
    )

layout = dict(title = "Open prices of AAPL.O",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
asset1_df = market_train_df[(market_train_df['assetCode'] == 'AAPL.O') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values
    )

layout = dict(title = "Closing prices of AAPL.O",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
asset1_df = market_train_df[(market_train_df['assetCode'] == 'AAPL.O') & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['volume'].values
    )

layout = dict(title = "Volume of AAPL.O",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )
data = [trace1]
py.iplot(dict(data=data, layout=layout), filename='basic-line')
# data visualization
import matplotlib.pyplot as plt
import seaborn as sns # advanced vizs
%matplotlib inline

# statistics
from statsmodels.distributions.empirical_distribution import ECDF

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# prophet by Facebook
from fbprophet import Prophet
sns.set(style = "ticks")# to format into seaborn 
c = '#386B7F' # basic color for plots
plt.figure(figsize = (12, 6))

plt.subplot(311)
cdf = ECDF(market_train_df['volume'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel(''); plt.ylabel('ECDF');

# plot second ECDF  
plt.subplot(312)
cdf = ECDF(market_train_df['open'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('');plt.ylabel('ECDF');

# plot second ECDF  
plt.subplot(313)
cdf = ECDF(market_train_df['close'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('');plt.ylabel('ECDF');
market_train_df[(market_train_df.assetCode=='A.N')].head()
voluem_an = np.transpose(pd.DataFrame([(market_train_df[(market_train_df.assetCode=='A.N')]).set_index('date')['volume']]))
voluem_an = np.transpose(pd.DataFrame([(market_train_df[(market_train_df.assetCode=='A.N')]).set_index('date')['volume']]))
open_an = np.transpose(pd.DataFrame([(market_train_df[(market_train_df.assetCode=='A.N')]).set_index('date')['open']]))
close_an = np.transpose(pd.DataFrame([(market_train_df[(market_train_df.assetCode=='A.N')]).set_index('date')['close']]))

f, (ax1, ax2, ax3) = plt.subplots(3, figsize = (12, 13))

# store types
voluem_an.resample('W').sum().plot(color = c, ax = ax1)
open_an.resample('W').sum().plot(color = c, ax = ax2)
close_an.resample('W').sum().plot(color = c, ax = ax3)
f, (ax1, ax2, ax3) = plt.subplots(3, figsize = (12, 13))

# monthly
decomposition_a = seasonal_decompose(voluem_an, model = 'additive', freq = 365)
decomposition_a.trend.plot(color = c, ax = ax1)

decomposition_b = seasonal_decompose(open_an, model = 'additive', freq = 365)
decomposition_b.trend.plot(color = c, ax = ax2)

decomposition_c = seasonal_decompose(close_an, model = 'additive', freq = 365)
decomposition_c.trend.plot(color = c, ax = ax3)
# figure for subplots
plt.figure(figsize = (12, 8))

# acf and pacf for volume
plt.subplot(321); plot_acf(voluem_an, lags = 50, ax = plt.gca(), color = c)
plt.subplot(322); plot_pacf(voluem_an, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for open
plt.subplot(323); plot_acf(open_an, lags = 50, ax = plt.gca(), color = c)
plt.subplot(324); plot_pacf(open_an, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for close
plt.subplot(325); plot_acf(close_an, lags = 50, ax = plt.gca(), color = c)
plt.subplot(326); plot_pacf(close_an, lags = 50, ax = plt.gca(), color = c)

plt.show()
df = market_train_df[(market_train_df["assetCode"] == 'A.N')]

volume = df.loc[:, ['date', 'volume']]

# reverse to the order: from 2013 to 2015
volume = volume.sort_index(ascending = True)

# to datetime64
volume['date'] = pd.DatetimeIndex(volume['date'])
volume.dtypes

# from the prophet documentation every variables should have specific names
volume = volume.rename(columns = {'date': 'ds',
                                'volume': 'y'})
volume.head()
# plot daily sales
ax = volume.set_index('ds').plot(figsize = (12, 4), color = c)
ax.set_ylabel('Daily volume of A.N')
ax.set_xlabel('Date')
plt.show()
# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width = 0.95)
my_model.fit(volume)

# dataframe that extends into future 6 weeks 
future_dates = my_model.make_future_dataframe(periods = 1)

print("First day to forecast.")
future_dates

# predictions
forecast = my_model.predict(future_dates)

# preditions for last week
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})
my_model.plot(forecast);
my_model.plot_components(forecast);