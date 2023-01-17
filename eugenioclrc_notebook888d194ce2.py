# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





#define a conversion function for the native timestamps in the csv file

def dateparse (time_in_secs):    

    return datetime.datetime.fromtimestamp(float(time_in_secs))



data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20', parse_dates=True, date_parser=dateparse, index_col=[0])



# Any results you write to the current directory are saved as output.
# First thing is to fix the data for bars/candles where there are no trades. 

# Volume/trades are a single event so fill na's with zeroes for relevant fields...

data['Volume_(BTC)'].fillna(value=0, inplace=True)

data['Volume_(Currency)'].fillna(value=0, inplace=True)

data['Weighted_Price'].fillna(value=0, inplace=True)



# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so

# lets fill forwards those values...

data['Close'].fillna(method='ffill', inplace=True)
## candle stick of 2hs

data['emah12'] = pd.Series(data['Close']).rolling(window=12 * 2 * 60).mean()

data['emah26'] = pd.Series(data['Close']).rolling(window=26 * 2 * 60).mean()

data['emah200'] = pd.Series(data['Close']).rolling(window=200 * 2 * 60).mean()
data['Buy'] = np.zeros(len(data))

data['Sell'] = np.zeros(len(data))



n = 0

prevrow = {'emah12': 0, 'emah26': 0}

log = []

lastaction = ''

for index, row in data.iterrows():

    prev12 = prevrow['emah12']

    prev26 = prevrow['emah26']

    prevrow = row

    if math.isnan(row['emah200']):

        continue

    if (row['emah200'] >= row['emah26'] or row['emah200'] >= row['emah12']):

        continue

    n+=1

    

    action = ''

    if (prev12 < prev26 and row['emah12'] > row['emah26']):

        action = 'Sell'

    if (prev12 > prev26 and row['emah12'] < row['emah26']):

        action = 'Buy'

    

    if (action and action != lastaction):

        if (action == 'Buy'):

            row[action] = 1

        else:

            row[action] = -1

        # print(action)

        lastaction = action

        log.append([action, row])  

# lets now take a look and see if its doing something sensible

import matplotlib

import matplotlib.pyplot as plt



fig,ax1 = plt.subplots(1,1)

ax1.plot(data['Close'])

y = ax1.get_ylim()

ax1.set_ylim(y[0] - (y[1]-y[0])*0.4, y[1])



ax2 = ax1.twinx()

ax2.set_position(matplotlib.transforms.Bbox([[0.125,0.1],[0.9,0.32]]))

ax2.plot(data['Buy'], color='#77dd77')

ax2.plot(data['Sell'], color='#dd4444')
j = 0;

ok = 0

bad = 1



for a in log:

    if (a[0] =='Buy'):

        buyp = a[1]['Close']

    else:

        sellp = a[1]['Close']

        

        if ((sellp - buyp) > 0):

            ok += 1

        else:

            bad += 1

        print((sellp - buyp)/sellp)

    j += 1



print("bad", bad)

print("ok", ok)