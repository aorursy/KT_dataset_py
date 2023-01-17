import datetime

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import pandas_datareader.data as pdr_data

from sklearn.preprocessing import MinMaxScaler
# TULIP BUBBLE

# Source: http://arno.uvt.nl/show.cgi?fid=129437

# This data are shifted forward by 100 years, because pandas doesn't handle

# such old timestamps.

data = [

    { 'date': pd.Timestamp('17341201'), 'price': 22 },

    { 'date': pd.Timestamp('17360501'), 'price': 61 },

    { 'date': pd.Timestamp('17360601'), 'price': 38 },

    { 'date': pd.Timestamp('17360701'), 'price': 51 },

    { 'date': pd.Timestamp('17360801'), 'price': 61 },

    { 'date': pd.Timestamp('17361101'), 'price': 97 },

    { 'date': pd.Timestamp('17361201'), 'price': 176 },

    { 'date': pd.Timestamp('17370201'), 'price': 199 },

    { 'date': pd.Timestamp('17370203'), 'price': 202 },

    { 'date': pd.Timestamp('17370205'), 'price': 178 },

    { 'date': pd.Timestamp('17370209'), 'price': 148 },

    { 'date': pd.Timestamp('17370211'), 'price': 145 },

    { 'date': pd.Timestamp('17370501'), 'price': 11 },

#     { 'date': pd.Timestamp('17420101'), 'price': 30 },

]

tulip = pd.DataFrame(data)

tulip.index = tulip['date']

tulip.drop(['date'], 1, inplace=True)
# THE SOUTH SEAS BUBBLE

south_sea = pd.read_csv("../input/south-sea-bubble/South_Seas_Bubble_1.csv", header=None, names=['date', 'price'])

south_sea['date'] = pd.to_datetime(south_sea['date'], dayfirst=True) 

south_sea.index = south_sea['date']

south_sea.drop(['date'], 1, inplace=True)
# JAPAN'S REAL ESTATE AND STOCK MARKET BUBBLE
# THE DOT-COM BUBBLE

dot_com = pd.read_csv('../input/nasdaqcomposite/IXIC.csv')

dot_com['Date'] = pd.to_datetime(dot_com['Date']) 

dot_com.index = dot_com['Date']

dot_com.drop(['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], 1, inplace=True)

dot_com.columns = ['price']
# THE US HOUSING BUBBLE
# BITCOIN

cryptos = pd.read_csv('../input/all-crypto-currencies/crypto-markets.csv')

bitcoin = cryptos[cryptos['symbol'] == 'BTC']

bitcoin.index = bitcoin['date']

bitcoin.drop(['slug', 'symbol', 'name', 'ranknow', 'date', 'open', 'high', 'low', 'volume', 'market', 'close_ratio', 'spread'], 1, inplace=True)

bitcoin.columns = ['price']
# AMAZON current price

stocks = pd.read_csv('../input/sandp500/all_stocks_5yr.csv')

amazon = stocks[stocks['Name'] == 'AMZN']

amazon['date'] = pd.to_datetime(amazon['date']) 

amazon.index = amazon['date']

amazon.drop(['date', 'open', 'high', 'low', 'volume', 'Name'], 1, inplace=True)

amazon.columns = ['price']
# DDD

# ddd = stocks[stocks['Name'] == 'DDD']

# ddd['date'] = pd.to_datetime(ddd['date']) 

# ddd.index = ddd['date']

# print(ddd.tail(3))

# ddd.drop(['date', 'open', 'high', 'low', 'volume', 'Name'], 1, inplace=True)

# ddd.columns = ['price']

# stocks['Name'].unique()
# Recreate the data frames based on a daily price, and use a number of days as index, so we 

# can bring all the bubbles together. 

# Also normalize the prices between 0 and 1. I ignore the relative amplitude of the bubbles.

def normalize(df):

    x = df[['price']].values.astype(float)

    min_max_scaler = MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)

    return pd.DataFrame(x_scaled)



def reindex_day(df):

    out = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='1D'))      

    out = out.interpolate(method='time')

    out.reset_index(inplace=True)

    out.drop(['index'], 1, inplace=True)

    return normalize(out)



tulip_day = reindex_day(tulip)

south_sea_day = reindex_day(south_sea)

dot_com_mask = (dot_com.index > '1998-1-1') & (dot_com.index <= '2003-1-1')

dot_com_day = reindex_day(dot_com[dot_com_mask])

bitcoin_mask = (bitcoin.index > '2016-1-1')

bitcoin_day = reindex_day(bitcoin[bitcoin_mask])

bitcoin_last_day = reindex_day(bitcoin[(bitcoin.index > '2012-1-1') & (bitcoin.index <= '2015-1-1')])

amazon_day = reindex_day(amazon[(amazon.index > '2012-1-1')])

# ddd_day = reindex_day(ddd[(ddd.index > '2011-1-1') & (ddd.index <= '2017-1-1')])
def get_paddings(*args):

    maxima = []

    for var in args:

        maxima.append(var.idxmax().item())

    return max(maxima) - np.array(maxima)

    

pads = get_paddings(tulip_day, south_sea_day, dot_com_day, bitcoin_day, bitcoin_last_day, amazon_day)
# Plots

fig, ax = plt.subplots(1, figsize=(14, 5))

ax.plot(np.arange(pads[0], pads[0]+len(tulip_day)),tulip_day, '-', label='tulip')

ax.plot(np.arange(pads[1], pads[1]+len(south_sea_day)), south_sea_day, '-', label='south seas')

ax.plot(np.arange(pads[2], pads[2]+len(dot_com_day)), dot_com_day, '-', label='nasdaq composite')

ax.plot(np.arange(pads[3], pads[3]+len(bitcoin_day)), bitcoin_day, '-', label='bitcoin')

ax.plot(np.arange(pads[4], pads[4]+len(bitcoin_last_day)), bitcoin_last_day, '-', label='bitcoin last bubble')

ax.plot(np.arange(pads[5], pads[5]+len(amazon_day)), amazon_day, '-', label='AMZN')

# ax.plot(np.arange(pads[6], pads[6]+len(ddd_day)), ddd_day, '-', label='DDD')

plt.legend()

plt.show()
# Include Amazon price to compare (or maybe the NASDAQ, 500, etc...). We are maybe in another stock bubble too?