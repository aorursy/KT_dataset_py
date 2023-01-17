import matplotlib.pyplot as plt

import matplotlib

import plotly.plotly as py

import datetime as dt

import matplotlib.dates as mdates 

plt.style.use('fivethirtyeight')

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

import seaborn as sns 

import numpy as np

import pandas as pd

import numpy as np

import random as rnd

import re

import io

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler

from numpy import genfromtxt

from scipy.stats import multivariate_normal

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score , average_precision_score

from sklearn.metrics import precision_score, precision_recall_curve

%matplotlib inline



from mpl_toolkits.basemap import Basemap

from matplotlib import animation, rc

from IPython.display import HTML



import warnings

warnings.filterwarnings('ignore')



import base64

from IPython.display import HTML, display

import warnings

warnings.filterwarnings('ignore')

from scipy.misc import imread

import codecs
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
def readabledate (unixtime):    

    return dt.datetime.fromtimestamp(float(unixtime))



Bitcoin = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv',parse_dates=[0], date_parser=readabledate)

EthereumPrice = pd.read_csv('../input/ethereum-historical-data/EtherMarketCapChart.csv',parse_dates=[1], date_parser=readabledate)

EthereumAddress = pd.read_csv('../input/ethereum-historical-data/EthereumUniqueAddressGrowthRate.csv',parse_dates=[1], date_parser=readabledate)

Allcrypto = pd.read_csv('../input/all-crypto-currencies/crypto-markets.csv')

name = Allcrypto['name'].unique()



Feature = []

currency  = []

marketval = []

v_features = name[:10]

for i, cn in enumerate(v_features):

     Feature.append(str(cn)) 

     filtered = Allcrypto[(Allcrypto['name']==str(cn))]

     temp = filtered[filtered['market'] == filtered['market'].max()]['name'].values

     temp1 = temp[0]

     tempval = filtered['market'].max()

     currency.append(temp1)

     marketval.append(tempval)



f, ax = plt.subplots(figsize=(13, 8)) 

g = sns.barplot( y = Feature,

            x = marketval,

                palette="summer")

plt.title("Top 10 Cryptocurrencies in the market")

ax.set_xticklabels(ax.get_xticks())

ax.get_yaxis().set_visible(False)

for i, v in enumerate(currency): 

    ax.text(2500000000, i, v,fontsize=18,color='brown',weight='bold')

fig=plt.gcf()

plt.show()
Bitcoin.tail(5)
years = np.unique(Bitcoin['Timestamp'].dt.year)

mean_open = []

mean_volume = []

mean_close = []

mean_high = []

mean_low = []

mean_BTC = []

mean_average = []

for year in years:

    mean_volume.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Volume_(Currency)'].mean())

    mean_BTC.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Volume_(BTC)'].mean())

    mean_open.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Open'].mean())

    mean_close.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Close'].mean())

    mean_high.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['High'].mean())

    mean_low.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Low'].mean())

    mean_average.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Weighted_Price'].mean())
trace0 = go.Scatter(

    x = years, 

    y = mean_average,

    #fill='tonexty',

    mode='lines',

    name='Weighted_Price of 1 Bitcoin',

    line=dict(

        color='rgb(199, 121, 093)',

    )

)



data = [trace0]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='Bitcoin price in $'),

    title=' Bitcoin value over the years ',

    showlegend = True)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
Months = {

1 :'January',

2 :'February',

3 :'March',

 4 :'April',

5 :'May',

 6 :'June',

 7 :'July',

8 :'August',

9 :'September',

10 :'October',

11 :'November',

12 :'December',

}



Traces = {

1 :'trace0',

2 :'trace1',

3 :'trace2',

 4 :'trace3',

5 :'trace4',

 6 :'trace5',

 7 :'trace6',

8 :'trace7',

9 :'trace8',

10 :'trace9',

11 :'trace10',

12 :'trace11',

}



DF_Bitcoin_price = {

1 : 'Bitcoin_price1',

2 : 'Bitcoin_price2',

3 : 'Bitcoin_price3',

4 : 'Bitcoin_price4',

5 : 'Bitcoin_price5',

6 : 'Bitcoin_price6',

7 : 'Bitcoin_price7',

8 : 'Bitcoin_price8',

9 : 'Bitcoin_price9',

10 : 'Bitcoin_price10',

11 : 'Bitcoin_price11',

12 : 'Bitcoin_price12'

}
Bitcoin['Century'] = Bitcoin['Timestamp'].dt.year

j = 1

data = []

for i in range(12):

     Bitcoin_month = Bitcoin[Bitcoin['Timestamp'].dt.month == j]

     DF_Bitcoin_price[j] = Bitcoin_month.groupby(['Century'])['Weighted_Price'].mean()

     Traces[j] = go.Scatter(

         x = DF_Bitcoin_price[j].index,

         y = DF_Bitcoin_price[j].values,

         mode = 'lines',

         name = Months[j]

     )

     data.append(Traces[j]) 

     j = j + 1



layout = go.Layout(

      xaxis=dict(title='year'),

      yaxis=dict(title='Bitcoin Price$$'),

      title=('Monthly distribution of Bitcoin prices'))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
years = np.unique(Bitcoin['Timestamp'].dt.year)

sum_coins = []

sum_volume = []

for year in years:

    sum_volume.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Volume_(Currency)'].sum())

    sum_coins.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Volume_(BTC)'].sum())

trace0 = go.Scatter(

    x = years, 

    y = sum_volume,

    fill= None,

    mode='lines',

    name='Trade volume in $ yearwise',

    line=dict(

        color='rgb(0, 255, 255)',

    )

)

data = [trace0]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='Trade volume in $'),

    title='Trade volume in $ yearwise',

    showlegend = True)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace0 = go.Scatter(

    x = years, 

    y = sum_coins,

    #fill='tonexty',

    mode='lines',

    name='Bitcoin mined /used for trading yearwise',

    line=dict(

        color='rgb(199, 121, 093)',

    )

)



data = [trace0]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='Bitcoins'),

    title='Bitcoins mined /used for trading yearwise',

    showlegend = True)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
Bitcoin_2017 = Bitcoin[Bitcoin['Timestamp'].dt.year == 2017]

months = Bitcoin_2017.groupby(Bitcoin_2017['Timestamp'].dt.month)['Volume_(BTC)'].sum().keys() 

months = np.array(months)

fig = plt.figure(figsize=(12,6))

axes1 = fig.add_axes([0.1,0.1,0.9,0.9])

axes2 = fig.add_axes([0.15,0.65,0.2,0.3])

axes3 = fig.add_axes([0.45,0.65,0.2,0.3])

axes4 = fig.add_axes([0.75,0.65,0.2,0.3])

axes5 = fig.add_axes([0.15,0.2,0.2,0.3])

axes6 = fig.add_axes([0.45,0.2,0.2,0.3])

axes7 = fig.add_axes([0.75,0.2,0.2,0.3])



#axes1.plot(years,sum_volume)

axes1.set_title('')



axes2.plot(months, Bitcoin_2017.groupby(Bitcoin_2017['Timestamp'].dt.month)['Volume_(BTC)'].sum().values, color="Blue", lw=5);

axes2.set_title('bitcoins mined/used in Year 2017')



axes3.plot(Bitcoin_2017['Timestamp'].dt.month,Bitcoin_2017['Weighted_Price'],color='Gold')

axes3.set_title('bitcoin prices in Year 2017')



axes4.plot(months, Bitcoin_2017.groupby(Bitcoin_2017['Timestamp'].dt.month)['Volume_(Currency)'].sum().values, color="Red", lw=5);

axes4.set_title('Trade volume $ in Year 2017');

axes4.set_yticklabels(axes4.get_yticks());



axes5.plot(years,sum_coins , color="Blue", lw=5);

axes5.set_title('bitcoins mined/used over the years')

axes5.set_yticklabels(axes5.get_yticks())



axes6.plot(years,mean_average,color='Gold')

axes6.set_title('bitcoin prices over the years')



axes7.plot(years,sum_volume , color="Red", lw=5);

axes7.set_title('Trade volume over the years');

axes7.set_yticklabels(axes7.get_yticks());



axes1.set_xticks([])

axes1.set_yticks([]);
EthereumPrice.tail(5)
trace0 = go.Scatter(

    x = EthereumPrice['UnixTimeStamp'], 

    y = EthereumPrice['Price'],

    #fill='tonexty',

    mode='lines',

    name='Price of Ethereum',

    line=dict(

        color='rgb(199, 121, 093)',

    )

)



data = [trace0]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='Ethereum price in $'),

    title=' Ethereum price over the years ',

    showlegend = True)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace0 = go.Scatter(

    x = EthereumPrice['UnixTimeStamp'], 

    y = EthereumPrice['Supply'],

    #fill='tonexty',

    mode='lines',

    name='supply of Ethereum coins',

    line=dict(

        color='rgb(199, 121, 093)',

    )

)



data = [trace0]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='Supply of Ethereum coins'),

    title='supply of Ethereum coins over the years ',

    showlegend = True)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace0 = go.Scatter(

    x = EthereumPrice['UnixTimeStamp'], 

    y = EthereumPrice['MarketCap'],

    #fill='tonexty',

    mode='lines',

    name='MarketCap in Million',

    line=dict(

        color='rgb(199, 121, 093)',

    )

)



data = [trace0]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='MarketCap in Million $'),

    title='MarketCap of Ethereum over the years ',

    showlegend = True)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace0 = go.Scatter(

    x = EthereumAddress['UnixTimeStamp'], 

    y = EthereumAddress['Value'],

    #fill='tonexty',

    mode='lines',

    name='No of Address',

    line=dict(

        color='rgb(199, 121, 093)',

    )

)



data = [trace0]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='No of Address'),

    title='No of growing Addresses over the years ',

    showlegend = True)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
hfmt = mdates.DateFormatter('%Y-%m')

hfmt2017 = mdates.DateFormatter('%m-%d')
EthereumPrice_2017 = EthereumPrice[EthereumPrice['UnixTimeStamp'].dt.year == 2017]



fig = plt.figure(figsize=(12,6))

axes1 = fig.add_axes([0.1,0.1,0.9,0.9])

axes2 = fig.add_axes([0.15,0.65,0.25,0.3])

axes3 = fig.add_axes([0.45,0.65,0.25,0.3])

axes4 = fig.add_axes([0.75,0.65,0.25,0.3])

axes5 = fig.add_axes([0.15,0.2,0.25,0.3])

axes6 = fig.add_axes([0.45,0.2,0.25,0.3])

axes7 = fig.add_axes([0.75,0.2,0.25,0.3])



#axes1.plot(years,sum_volume)

axes1.set_title('')



axes2.plot(EthereumPrice_2017['UnixTimeStamp'],EthereumPrice_2017['Supply'] , color="Blue", lw=5);

axes2.set_title('Ethereum coins supply in Year 2017')

axes2.set_yticklabels(axes2.get_yticks())

axes2.set_xticklabels(axes2.get_xticks(),rotation=70)

axes2.xaxis.set_major_formatter(hfmt2017)



axes3.plot(EthereumPrice_2017['UnixTimeStamp'],EthereumPrice_2017['Price'],color='Gold')

axes3.set_xticklabels(axes3.get_xticks(),rotation=70)

axes3.xaxis.set_major_formatter(hfmt2017)

axes3.set_title('Ethereum price in Year 2017')



axes4.plot(EthereumPrice_2017['UnixTimeStamp'],EthereumPrice_2017['MarketCap'], color="Red", lw=5);

axes4.set_title('Market cap in Year 2017');

axes4.set_xticklabels(axes4.get_xticks(),rotation=70);

axes4.xaxis.set_major_formatter(hfmt2017)

axes4.set_yticklabels(axes4.get_yticks());



axes5.plot(EthereumPrice['UnixTimeStamp'],EthereumPrice['Supply'], color="Blue", lw=5);

axes5.set_title('Ethereum coins supply over the years')

axes5.set_yticklabels(axes5.get_yticks())

axes5.set_xticklabels(axes5.get_xticks(),rotation=70)

axes5.xaxis.set_major_formatter(hfmt)



axes6.plot(EthereumPrice['UnixTimeStamp'],EthereumPrice['Price'],color='Gold')

axes6.set_title('Ethereum price over the years')

axes6.set_xticklabels(axes6.get_xticks(),rotation=70)

axes6.xaxis.set_major_formatter(hfmt)



axes7.plot(EthereumPrice['UnixTimeStamp'],EthereumPrice['MarketCap'], color="Red", lw=5);

axes7.set_title('Market cap over the years');

axes7.set_xticklabels(axes7.get_xticks(),rotation=70)

axes7.xaxis.set_major_formatter(hfmt)

axes7.set_yticklabels(axes7.get_yticks());



axes1.set_xticks([])

axes1.set_yticks([]);
Allcrypto.head()
Bitcoin = Allcrypto[Allcrypto['ranknow'] == 1]

Ethereum = Allcrypto[Allcrypto['ranknow'] == 2]

Ripple  = Allcrypto[Allcrypto['ranknow'] == 3]

from matplotlib.finance import candlestick_ohlc

BitcoinOHLC = Bitcoin[['date','open','high','low','close']]

EthereumOHLC = Ethereum[['date','open','high','low','close']]

RippleOHLC = Ripple[['date','open','high','low','close']]

import matplotlib.dates as mdates

BitcoinOHLC['date'] = pd.to_datetime(BitcoinOHLC['date'])

BitcoinOHLC['date'] = mdates.date2num(BitcoinOHLC['date'].astype(dt.date))

RippleOHLC['date'] = pd.to_datetime(RippleOHLC['date'])

RippleOHLC['date'] = mdates.date2num(RippleOHLC['date'].astype(dt.date))

EthereumOHLC['date'] = pd.to_datetime(EthereumOHLC['date'])

EthereumOHLC['date'] = mdates.date2num(EthereumOHLC['date'].astype(dt.date))
f,ax=plt.subplots(figsize=(15,11))

ax.xaxis_date()

plt.xlabel("Date")

candlestick_ohlc(ax,BitcoinOHLC.values,width=5, colorup='g', colordown='r',alpha=0.75)

ax.set_xticklabels(ax.get_xticks(),rotation=70)

ax.xaxis.set_major_formatter(hfmt)

plt.ylabel("Price")

plt.legend()

plt.show()
f,ax=plt.subplots(figsize=(15,11))

ax.xaxis_date()

plt.xlabel("Date")

candlestick_ohlc(ax,EthereumOHLC.values,width=5, colorup='g', colordown='r',alpha=0.75)

ax.set_xticklabels(ax.get_xticks(),rotation=70)

ax.xaxis.set_major_formatter(hfmt)

plt.ylabel("Price")

plt.legend()

plt.show()
EthereumOHLC = EthereumOHLC[EthereumOHLC['date'] > 736630.0]

BitcoinOHLC = BitcoinOHLC[BitcoinOHLC['date'] > 736630.0]



fig = plt.figure(figsize=(12,12))

axes1 = fig.add_axes([0.1,0.1,0.9,0.9])

axes2 = fig.add_axes([0.15,0.65,0.8,0.3])

axes3 = fig.add_axes([0.15,0.2,0.8,0.3])



axes2.xaxis_date()

candlestick_ohlc(axes2,BitcoinOHLC.values,width=2, colorup='g', colordown='r',alpha=0.70)

axes2.set_xticklabels(axes2.get_xticks(),rotation=70)

axes2.xaxis.set_major_formatter(hfmt)

axes2.set_title('Bitcoin Candlestick chart for last few months of 2017');

#axes2.scatter(736688.0, 17000,marker="o", color="white", s=50000, linewidths=0)

axes2.annotate('Bearish Engulfing', (736684.0,19000),fontsize=14,rotation=0,color='r')

axes2.annotate('.', xy=(736684.0,17000), xytext=(736688.0, 19000),

            arrowprops=dict(facecolor='Red', shrink=0.06),

            )

axes2.annotate('.', xy=(736691.0,16000), xytext=(736689.0, 19000),

            arrowprops=dict(facecolor='Red', shrink=0.06),

            )





axes3.xaxis_date()

candlestick_ohlc(axes3,EthereumOHLC.values,width=2, colorup='g', colordown='r',alpha=0.70)

axes3.set_xticklabels(axes3.get_xticks(),rotation=70)

axes3.xaxis.set_major_formatter(hfmt)

axes3.set_title('Ethereum Candlestick chart for last few months of 2017');

axes3.annotate('Bullish Engulfing', (736684.0,1300),fontsize=14,rotation=0,color='g')

axes3.annotate('.', xy=(736695.0,850), xytext=(736688.0, 1300),

            arrowprops=dict(facecolor='Green', shrink=0.06),

            )





axes1.set_xticks([])

axes1.set_yticks([]);
