import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import plotly as pt

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



warnings.filterwarnings('ignore')



print('*'*50)

#print('Python Version    : ', sys.version)

print('Pandas Version    : ', pd.__version__)

print('Numpy Version     : ', np.__version__)

print('Matplotlib Version: ', mpl.__version__)

print('Seaborn Version   : ', sns.__version__)

print('Plotly Version    : ', pt.__version__)

print('*'*50)
def convertCurrency(val):

    """

    Convert the string number value to a float value

     - Remove commas

     - Convert to float type

    """

    new_val = val.replace(',','')

    return float(new_val)



def convertVolume(val):

    """

    Convert the string to an actual floating point value

    - Remove M

    - Remove K

    """

    new_val = val.replace('M', '').replace('K','')

    return float(new_val)



def convertPercent(val):

    """

    Convert the percentage string to an actual floating point percent

    - Remove %

    - Divide by 100 to make decimal

    """

    new_val = val.replace('%', '')

    return float(new_val) / 100
iotaUSD = pd.read_csv("../input/IOT_USD Bitfinex Historical Data.csv")

iotaBTC = pd.read_csv("../input/IOT_BTC Bitfinex Historical Data.csv")

btcUSD  = pd.read_csv("../input/BTC_USD Bitfinex Historical Data.csv")
#Reverse the index

iotaUSD = iotaUSD.sort_index(axis=0 ,ascending=False)

iotaBTC = iotaBTC.sort_index(axis=0 ,ascending=False)

btcUSD = btcUSD.sort_index(axis=0 ,ascending=False)
#change the datetime col format 

iotaUSD['Date'] = pd.to_datetime(iotaUSD['Date'])

iotaBTC['Date'] = pd.to_datetime(iotaBTC['Date'])

btcUSD['Date'] = pd.to_datetime(btcUSD['Date'])
#reindexing by datetime

iotaUSD.index = iotaUSD['Date']

iotaBTC.index = iotaBTC['Date']

btcUSD.index = btcUSD['Date']
print('Shape IOTA-USD:', iotaUSD.shape)

print('Shape IOTA-BTC:', iotaBTC.shape)

print('Shape BTC-USD :', btcUSD.shape)
iotaUSD.head(3)
iotaUSD.info()
iotaUSD['Vol.'] = iotaUSD['Vol.'].apply(convertVolume)

iotaUSD['Change %'] = iotaUSD['Change %'].apply(convertPercent)
iotaUSD.info()
iotaUSD.describe()
iotaBTC.head(3)
iotaBTC.info()
iotaBTC['Vol.'] = iotaBTC['Vol.'].apply(convertVolume)

iotaBTC['Change %'] = iotaBTC['Change %'].apply(convertPercent)
iotaBTC.describe()
btcUSD.head(3)
btcUSD.info()
toConvert = ['Price', 'Open', 'High', 'Low']

for col in toConvert:

    btcUSD[col] = btcUSD[col].apply(convertCurrency)

    

btcUSD['Vol.'] = btcUSD['Vol.'].apply(convertVolume)

btcUSD['Change %'] = btcUSD['Change %'].apply(convertPercent)
btcUSD.describe()
plt.figure(figsize=(20,10))

iotaUSD['Price'].plot(linewidth = 3)

plt.ylabel('Value in USD', fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.xticks(fontsize=15)

plt.title('IOTA-USD', fontsize=20)

plt.show()
plt.figure(figsize=(30,24))

sns.set_style('white')



plt.subplot(311)

iotaUSD['Price'].plot(linewidth = 3)

plt.ylabel('Value in USD', fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.xticks(fontsize=15)

plt.title('IOTA-USD', fontsize=20)



plt.subplot(312)

btcUSD['Price'].plot(linewidth = 3, color='k')

plt.ylabel('Value in USD', fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.xticks(fontsize=15)

plt.title('BTC-USD', fontsize=20)



plt.subplot(313)

iotaBTC['Price'].plot(linewidth = 3, color='r')

plt.ylabel('Value in BTC', fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.xticks(fontsize=15)

plt.title('IOTA-BTC', fontsize=20)



plt.show()
plt.figure(figsize=(20,10))

iotaUSD['Vol.'].plot(linewidth = 2)

plt.ylabel('Volume in millions USD', fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.xticks(fontsize=15)

plt.title('IOTA Traded Volume in USD', fontsize=20)

plt.show()
minPrice = iotaUSD['Price'].min()

maxPrice = iotaUSD['Price'].max()

minData  = iotaUSD[iotaUSD['Price'] == minPrice]

maxData  = iotaUSD[iotaUSD['Price'] == maxPrice]



print('IOTA Historical MinPrice: ', minPrice)

print('IOTA Historical MaxPrice: ', maxPrice)
print('IOTA min price')

print(minData)
trace = go.Candlestick(x=iotaUSD['2017-07'].index,

                open=iotaUSD['2017-07'].Open,

                high=iotaUSD['2017-07'].High,

                low=iotaUSD['2017-07'].Low,

                close=iotaUSD['2017-07'].Price)



layout = go.Layout(

    title='IOTA Candlestick July 17',

    xaxis=dict(

        title='Time',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='Price USD',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )))

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='IOTA Candlestick July 17')
print('IOTA max price')

print(maxData)
trace1 = go.Candlestick(x=iotaUSD['2017-12'].index,

                open=iotaUSD['2017-12'].Open,

                high=iotaUSD['2017-12'].High,

                low=iotaUSD['2017-12'].Low,

                close=iotaUSD['2017-12'].Price)



layout = go.Layout(

    title='IOTA Candlestick December 17',

    xaxis=dict(

        title='Time',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='Price USD',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )))

data = [trace1]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='Candlestick IOTA December 17')
print('BITCOIN max price')

print(btcUSD[btcUSD['Price'] == btcUSD['Price'].max()])
# Candlestick chart of December 2017

trace0 = go.Candlestick(x=(btcUSD['2017-12'].index),

                open=btcUSD['2017-12'].Open,

                high=btcUSD['2017-12'].High,

                low=btcUSD['2017-12'].Low,

                close=btcUSD['2017-12'].Price)



layout = go.Layout(

    title='BITCOIN Candlestick December 17',

    xaxis=dict(

        title='Time',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='Price USD',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )))

data = [trace0]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='BITCOIN Candlestick December 17')
maxVol = iotaUSD['Vol.'].max()

maxVolume = iotaUSD[iotaUSD['Vol.'] == maxVol]

print('IOTA max daily volume')

print(maxVolume)
#IOTA Volume vs USD visualization

trace = go.Scattergl(

    y = iotaUSD['Vol.'].astype(float),

    x = iotaUSD['Price'].astype(float),

    mode = 'markers',

    marker = dict(

        color = '#00a1a1',

        line = dict(width = 1)))



layout = go.Layout(

    title='IOTA Volume vs USD',

    xaxis=dict(

        title='Close Price USD',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title=' Daily Volume IOTA',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )))

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='compareVolumeUsd')
meanByWeekday = iotaUSD.groupby(iotaUSD.index.dayofweek).mean()

medianByWeekday = iotaUSD.groupby(iotaUSD.index.dayofweek).median()

meanByWeekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

medianByWeekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
meanByWeekday
medianByWeekday
plt.figure(figsize=(12,5))

meanByWeekday['Price'].plot()

medianByWeekday['Price'].plot()

plt.ylabel('Price in USD', fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Day of Week', fontsize=15)

plt.xticks(fontsize=15, rotation=0)

plt.title('IOTA Closing Price per Day of Week',fontsize=20)

plt.legend(labels=['Mean Price', 'Median Price'])

plt.show()
plt.figure(figsize=(12,5))

meanByWeekday['Vol.'].plot()

medianByWeekday['Vol.'].plot()

plt.ylabel('Volume in millions USD', fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Day of Week', fontsize=15)

plt.xticks(fontsize=15, rotation=0)

plt.title('IOTA Volume per Day of Week',fontsize=20)

plt.legend(labels=['Mean Vol.', 'Median Vol.'])

plt.show()
plt.figure(figsize=(15,5))

iotaUSD['High'].plot(linewidth = 3)

plt.ylabel('Value in USD', fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.xticks(fontsize=15)

plt.title('Daily High value IOTA-USD', fontsize=20)

plt.show()
import statsmodels.api as sm

from pylab import rcParams



#DECOMPOSITION

rcParams['figure.figsize'] = 11, 9

decomposed_IOTA_volume = sm.tsa.seasonal_decompose(iotaUSD["High"],freq=360) #The frequency is annual

figure = decomposed_IOTA_volume.plot()

plt.show()