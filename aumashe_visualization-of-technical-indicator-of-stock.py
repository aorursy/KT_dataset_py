# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib



matplotlib.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

hist = pd.read_csv(filepath_or_buffer="../input/EW-MAX.csv",delim_whitespace=False,header=0,sep=',')

hist['Date'] = pd.to_datetime(hist['Date'])

#find day of week and convert the day into dummy features 5->101, 4->100...

hist['DayofWeek'] = hist['Date'].dt.dayofweek  # Day of week 0...4

hist['Dayofweek_Dummy0'] = hist.apply(lambda row: row['DayofWeek'] == 0 and 1 or 0, axis =1)

hist['Dayofweek_Dummy1'] = hist.apply(lambda row: row['DayofWeek'] == 1 and 1 or 0, axis =1)

hist['Dayofweek_Dummy2'] = hist.apply(lambda row: row['DayofWeek'] == 2 and 1 or 0, axis =1)

hist['Dayofweek_Dummy3'] = hist.apply(lambda row: row['DayofWeek'] == 3 and 1 or 0, axis =1)

hist['Dayofweek_Dummy4'] = hist.apply(lambda row: row['DayofWeek'] == 4 and 1 or 0, axis =1)





hist.describe()
hist.head(10)
# Plot Close and Adj_Close. They are exactly the same. 

hist.plot(y=['Close','Adj_Close'])
#delete Adj_Close as it is redundant

del hist['Adj_Close']

hist.isnull().sum()
hist.shape
# Closing price n periods ago

def prev_closing(values):

    return values[0]

hist['Close_1D'] = hist['Close'].rolling(window=2,center=False).apply(func=prev_closing)

hist['Close_3D'] = hist['Close'].rolling(window=3,center=False).apply(func=prev_closing)

hist['Close_12D'] = hist['Close'].rolling(window=12,center=False).apply(func=prev_closing)

#hist["Close_1D"] = pd.DataFrame(hist["Close"][1:]).reset_index()["Close"]



#highest of n days period

hist['Highest_10D'] = hist['High'].rolling(window=10,center=False).max()

hist['Lowest_10D'] = hist['Low'].rolling(window=10,center=False).min()



hist.head(20)

# Momentum (today close - the last ith day close )

hist["Momentum_1D"] = hist["Close"] - hist["Close_1D"]

hist["Momentum_3D"] = hist["Close"] - hist["Close_3D"]



# Rate of changes n days : ROC = (Most recent closing price - Closing price n periods ago) / Closing price n periods ago x 100

hist["ROC_3D"] = ( hist["Close"] - hist["Close_3D"] ) / hist["Close_3D"] * 100

hist["ROC_12D"] = ( hist["Close"] - hist["Close_12D"] ) / hist["Close_12D"] * 100



# Williams %R:(highest-closed)/(highest-lowest)*100

#hist["Williams_10D"] = [ (hist["High"][i:i+10].max() - hist["Close"][i])/(hist["High"][i:i+9].max() - hist["Low"][i:i+10].min()) for i in range(hist.shape[0])]

hist['Williams_10D'] = hist.apply(lambda row: 100 * (row['Highest_10D'] - row['Close']) / (row['Highest_10D'] - row['Lowest_10D']),axis=1)



# Relative Strength Index

# Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100

# Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)>0};

#        PriceDown(t)=-1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)<0};

def rsi(values):

    dUp = values[values>0].mean()

    dDown = -1*values[values<0].mean()

    return 100 * dUp / (dUp + dDown)

hist['RSI_6D'] = hist['Momentum_1D'].rolling(center=False,window=6).apply(rsi)

hist['RSI_12D'] = hist['Momentum_1D'].rolling(center=False,window=12).apply(rsi)



# Simple Moving Average : SUM(Close[i],N) / N

#hist["SMA_5D"] = [(hist["Close"][i:i+5]).mean() for i in range(hist.shape[0])]

hist['SMA_5D'] = hist['Close'].rolling(center=False,window=5).mean()

hist['SMA_10D'] = hist['Close'].rolling(center=False,window=10).mean()

hist['SMA_20D'] = hist['Close'].rolling(center=False,window=20).mean()

hist['SMA_30D'] = hist['Close'].rolling(center=False,window=30).mean()

hist['SMA_50D'] = hist['Close'].rolling(center=False,window=50).mean()



#EWMA 

hist['EWMA_12D'] = hist['Close'].ewm(span=12).mean()#pd.ewma(arg=hist['Close'],span=12)

hist['EWMA_26D'] = hist['Close'].ewm(span=26).mean()#pd.ewma(arg=hist['Close'],span=26)



#MACD 

hist['MACD'] = hist['EWMA_12D'] - hist['EWMA_26D']



hist.plot(x= 'Date',y=['Close','EWMA_12D','EWMA_26D'],figsize=(25,20))

hist.plot(x= 'Date',y=['RSI_12D',],figsize=(25,8))
# Target: Momentum_3D VS 0

def target(values):

    if(values[-1] - values[0]>0):

        return 1

    else:

        return -1

#hist['Target'] = hist['Close'].rolling(center=False,window=4).apply(target)

hist['For_target'] = [hist["Close"][i+4] - hist["Close"][i] if i+4 < hist.shape[0] else 0 for i,record in hist.iterrows()]

hist["Target"] = [1 if i+4 < hist.shape[0] and hist["Close"][i+4] - hist["Close"][i] > 0 else -1 for i,record in hist.iterrows()]
# fill NaN

#hist = hist.fillna(0)

hist.head(10)
# Histograms

hist[["Close","High","Low","Open","Volume","Momentum_1D","Momentum_3D","ROC_3D","ROC_12D","Williams_10D","RSI_6D","RSI_12D","SMA_5D","SMA_10D","SMA_20D","SMA_30D","Target"]].hist(figsize=(15,13))
names = ["Close","Volume","Momentum_1D","Momentum_3D","ROC_3D","ROC_12D","Williams_10D","RSI_12D","SMA_30D","Target"]

corr = hist[names].corr()
# Correlation Matrix Plot

fig = plt.figure(figsize=(15,13))

ax = fig.add_subplot(111)

cax = ax.matshow(corr, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,9,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)
from pandas.tools.plotting import scatter_matrix

#scatter_matrix(hist[['DayofWeek','Momentum_1D']],figsize=(15,13))

hist[['DayofWeek','Momentum_1D']].plot.hexbin(x='DayofWeek',y='Momentum_1D',gridsize=25,figsize=(10,8))
hist.plot.scatter(x='DayofWeek', y='Momentum_1D', s=hist['Momentum_1D']);
hist.to_csv("../input/feature_engineering.csv")