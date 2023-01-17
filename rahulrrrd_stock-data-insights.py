!pip install pmdarima
import pandas as pd 

import numpy as np



import os 

import glob



import matplotlib.pyplot as plt 

from matplotlib import style

style.use('seaborn')

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.offline as pyo

import plotly.express as px 

import seaborn as sb

from pylab import rcParams

rcParams['figure.figsize'] = 15, 8

pyo.init_notebook_mode()



from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from pmdarima.arima import auto_arima

from sklearn.metrics import mean_squared_error, mean_absolute_error

import math



import warnings 



warnings.filterwarnings("ignore")
DATA_FOLDER = '/kaggle/input/stock-market-wide-datasets/'
Q_DATA = glob.glob(DATA_FOLDER+'*Q')

AM_DATA = DATA_FOLDER+'AM'

EVENT = DATA_FOLDER+'event'

NEWS = DATA_FOLDER+'news'
df= pd.DataFrame()

for e,i in enumerate(Q_DATA):    

    data = pd.read_csv(i)

    data = data.query("ticker == 'FANG'")

    if data.shape[0] == 0:

        continue

    elif e!=0:

       df= pd.concat([df,data])

    else:

        df = data
df.head()
print(df['time'].max())

print(df['time'].min())
am_df = pd.read_csv(AM_DATA)

am_df = am_df.query("symbol=='FANG'")

am_df['time'] = pd.to_datetime(am_df['time'].str[:-3])

am_df = am_df.sort_values(['time']).reset_index(drop=True)

am_df.set_index('time', inplace=True)

am_df.head()
df['time'] = df['time'].str[:-3]

df['time'] = pd.to_datetime(df['time'])

df = df.sort_values(['time']).reset_index(drop=True)

df.set_index('time', inplace=True)

df.head()
df_int = df.resample('15min').mean().dropna()


fig = go.Figure(go.Scatter(x = df_int.index.to_series().dt.strftime('%Y-%m-%d, %H'), y=df_int['ask_price'], name='ask_price'))

fig.add_trace(go.Scatter(x = df_int.index.to_series().dt.strftime('%Y-%m-%d, %H'), y=df_int['bid_price'], name='bid_price'))

fig.update_layout(title='Ask and Bid price of FANG from 4 Aug 2020 - 18 Aug 2020', yaxis_title='Price')

fig.show()
am_df_int = am_df.resample('15min').mean().dropna()

fig = go.Figure(go.Scatter(x = am_df_int.index.to_series().dt.strftime('%Y-%m-%d, %H'), y=am_df_int['close_price'], name='FANG'))

fig.update_layout(title='FANG stock price thorughout August 2020')

fig.show()
df_interval = am_df_int['2020-08-10':'2020-08-15']

df_interval = df_interval.join(df_int)

df_interval.head()


fig = make_subplots(

    shared_xaxes=True,

    rows=10, cols=1,

    specs=[[{"rowspan":6}],

           [{}],

           [{}],

           [{}],

           [{}],

           [{}],

           [{"rowspan": 2}],

          [{}],

          [{"rowspan":2}],

          [{}]],

    subplot_titles=("Movement of FANG Stock thorugh out a week in 15 min time interval","","","","","","","","",""))



fig.append_trace(go.Scatter(x = df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'), y=df_interval['ask_price'], name='ask_price'),row=1,col=1)

fig.append_trace(go.Scatter(x = df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'),y=df_interval['bid_price'], name='bid_price'),row=1,col=1)

fig.append_trace(go.Scatter(x = df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'), y=df_interval['close_price'], name='close_price'),row=1,col=1)

fig.append_trace(go.Bar(x = df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'), y=df_interval['bid_size'], name='bid_size'),row=7,col=1)

fig.append_trace(go.Bar(x = df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'), y=df_interval['ask_size'], name='ask_size'),row=9,col=1)



fig.update_layout( yaxis_title='Price')

fig.show()
mu = df_interval['close_price'].mean()

var = df_interval['close_price'].var()

median = df_interval['close_price'].median()

stddev = df_interval['close_price'].std()

price_max = df_interval['close_price'].max()

price_low = df_interval['close_price'].min()

print("mean: ",mu)

print("Var: ",var)

print("median: ",median)

print("stddev: ",stddev)

print("Max: ",price_max)

print("Min: ",price_low)
#Bolinger bands



window = 15

df_interval['MA15'] = df_interval['close_price'].rolling(window=window).mean()

df_interval['STD15'] = df_interval['close_price'].rolling(window=window).std()

df_interval['upper'] = df_interval['MA15'] + df_interval['STD15'] * 2

df_interval['lower'] = df_interval['MA15'] - df_interval['STD15'] * 2
fig = make_subplots(

    shared_xaxes=True,

    rows=4,cols=1,

    specs = [

        [{'rowspan':3}],

        [{'rowspan':1}],

        [{}],

        [{}]

    ],

    subplot_titles= ("","","","Volume"))

fig.add_trace(go.Scatter(x= df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'),y=df_interval['upper'], name='UPPER BOUND'),row=1,col=1)

fig.add_trace(go.Scatter(x= df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'),y=df_interval['lower'], name='LOWER BOUND'),row=1,col=1)

fig.add_trace(go.Scatter(x= df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'),y=df_interval['close_price'], name='Price'),row=1,col=1)

fig.add_trace(go.Scatter(x= df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'),y=df_interval['MA15'], name='Price'),row=1,col=1)

fig.add_trace(go.Bar(x= df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'),y=df_interval['volume'], name='Volume'),row=4,col=1)



fig.update_layout(title='Bolinger Bands')

fig.show()
df_interval['pct_change'] = df_interval['close_price'].pct_change()

df_interval['lag'] = df_interval['close_price'].shift(12)

df_interval['lag_returns'] = (df_interval['close_price'] - df_interval['lag'])/df_interval['close_price']

fig = make_subplots(

    shared_xaxes=True,

    rows=2, cols=1,

    specs=[[{"rowspan":1}],

           [{"rowspan": 1}],],

    subplot_titles=("% change / simple returns","Same day returns"))



fig.append_trace(go.Scatter(x = df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'), y=df_interval['pct_change'], name='% change'),row=1,col=1)

fig.append_trace(go.Scatter(x = df_interval.index.to_series().dt.strftime('%Y-%m-%d, %H-%M'),y=df_interval['lag_returns'], name='lag returns'),row=2,col=1)

fig.show()
fig = make_subplots(

    rows=2, cols=2,

    specs=[[{"rowspan":1},{"colspan":1}],

           [{"rowspan": 1},{"colspan":1}],],

    subplot_titles=("% change / simple returns","Same day Returns","close price histogram","Volume"))



fig.add_trace(go.Histogram(x=df_interval['pct_change']),row=1,col=1)

fig.append_trace(go.Histogram(x=df_interval['lag_returns']),row=1,col=2)

fig.add_trace(go.Histogram(x=df_interval['close_price']),row=2,col=1)

fig.add_trace(go.Histogram(x=df_interval['volume']),row=2,col=2)



fig.show()
px.scatter(df_interval, x='ask_price',y='bid_price',color='lag_returns', title='Ask Price vs Bid price')
df_interval['diff_ask_bid'] = df_interval['ask_price']-df_interval['bid_price']
model = np.polyfit(df_interval.dropna()['diff_ask_bid'],y=df_interval.dropna()['lag_returns'],deg=1)

m,c =model
fig = go.Figure(px.scatter(df_interval.dropna(), x ='diff_ask_bid', y='lag_returns', title='Linear relationship between difference in ask and bid price vs Same day returns'))

fig.add_trace(go.Scatter(x=df_interval.dropna()['diff_ask_bid'], y=(m*df_interval.dropna()['diff_ask_bid'])+c, name='Linear Fit'))

fig.show()
#Test for staionarity

def test_stationarity(timeseries):

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:

    

    plt.plot(timeseries, color='blue',label='Original')

    plt.plot(rolmean, color='red', label='Rolling Mean')

    plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best') 

    plt.title('Rolling Mean and Standard Deviation')

    plt.show(block=False)

 

    print("Results of dickey fuller test")

    adft = adfuller(timeseries,autolag='AIC')



    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])

    for key,values in adft[4].items():

        output['critical value (%s)'%key] =  values

    print(output)

    

test_stationarity(df_interval['close_price'])
result = seasonal_decompose(df_interval['close_price'], model='multiplicative', freq = 15);

fig = plt.figure();

fig = result.plot() ;

fig.set_size_inches(16, 9);