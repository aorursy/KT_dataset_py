#import yfinance as yf   

import pandas as pd

import matplotlib.pyplot as plt

#plt.style.use('dark_background')



stocks = ['FB', 'AAPL', 'AMZN', 'GOOG', 'NFLX']

stocks_returns = ['FB_r', 'AAPL_r', 'AMZN_r', 'GOOG_r', 'NFLX_r']



data = pd.read_csv('../input/yfinancedata/yfinancedata.csv')
data = data.set_index(data['Date'])

data = data.drop(columns = 'Date')

data.head()
plt.figure(figsize = (16,8))

for st in stocks:

    plt.plot(data.index, data[st], label = st)

    plt.legend()
plt.figure(figsize = (16,8))

for st in stocks:

    plt.plot(data.index, data[st]/data[st][0], label = st)

    plt.legend()
for st,sr in zip(stocks, stocks_returns) :

    data[sr] = data[st].pct_change()

data.head()
data = data.drop(columns = stocks, axis = 0)

data = data.dropna()



data.head()
weights = {

    'FB' : 0.0806,

    'AAPL' : 0.5087,

    'AMZN' : 0.3082,

    'GOOG': 0.0804,

    'NFLX': 0.08039283355,

}



respective_weights = list(weights.values())



plt.figure(figsize = (16,8))

plt.ylabel("Weights")

plt.title("Free Float")

plt.bar(stocks, respective_weights)
for st,sr in zip(stocks, stocks_returns):

    data[st] = (data[sr]+1).cumprod()

    

data= data.drop(columns = stocks_returns, axis = 0)

data
indx = pd.DataFrame(index=data.index.copy())

for st, w in zip(stocks, respective_weights):

    indx[st] = data[st]*w

    

indx['index'] = indx['FB'] + indx['AAPL'] + indx['AMZN'] + indx['GOOG'] + indx['NFLX']

indx
plt.figure(figsize = (16,8))

plt.plot(indx.index, indx['index'],linewidth= 5, alpha = 1, label = 'Index')

plt.plot(indx.index, data['FB'], alpha = 0.5, label = 'FB')

plt.plot(indx.index, data['AAPL'], alpha = 0.5, label = 'AAPL')

plt.plot(indx.index, data['AMZN'], alpha = 0.5, label = 'AMZN')

plt.plot(indx.index, data['GOOG'], alpha = 0.5, label = 'GOOG')

plt.plot(indx.index, data['NFLX'], alpha = 0.5, label = 'NFLX')

plt.legend()