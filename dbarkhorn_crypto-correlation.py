import pandas as pd

from pandas.plotting import lag_plot

import numpy as np

import sklearn as sk

from sklearn import preprocessing as pr

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from scipy.signal import correlate

from scipy.stats.mstats import spearmanr

from statsmodels.tsa.stattools import acf, adfuller

from statsmodels.graphics.tsaplots import plot_pacf
crypto = {}



crypto['bitcoin'] = pd.read_csv('../input/cryptocurrencypricehistory/bitcoin_price.csv')

crypto['bitcoin_cash'] = pd.read_csv("../input/cryptocurrencypricehistory/bitcoin_cash_price.csv")

crypto['dash'] = pd.read_csv("../input/cryptocurrencypricehistory/dash_price.csv")

crypto['ethereum'] = pd.read_csv("../input/cryptocurrencypricehistory/ethereum_price.csv")

crypto['iota'] = pd.read_csv("../input/cryptocurrencypricehistory/iota_price.csv")

crypto['litecoin'] = pd.read_csv("../input/cryptocurrencypricehistory/litecoin_price.csv")

crypto['monero'] = pd.read_csv("../input/cryptocurrencypricehistory/monero_price.csv")

crypto['nem'] = pd.read_csv("../input/cryptocurrencypricehistory/nem_price.csv")

crypto['neo'] = pd.read_csv("../input/cryptocurrencypricehistory/neo_price.csv")

crypto['numeraire'] = pd.read_csv("../input/cryptocurrencypricehistory/numeraire_price.csv")

crypto['ripple'] = pd.read_csv("../input/cryptocurrencypricehistory/ripple_price.csv")

crypto['stratis'] = pd.read_csv("../input/cryptocurrencypricehistory/stratis_price.csv")

crypto['waves'] = pd.read_csv("../input/cryptocurrencypricehistory/waves_price.csv")
# For this analysis I will only be looking at closing price to make things more manageable

for coin in crypto:

    for column in crypto[coin].columns:

        if column not in ['Date', 'Close']:

            crypto[coin] = crypto[coin].drop(column, 1)

    # Make date the datetime type and reindex

    crypto[coin]['Date'] = pd.to_datetime(crypto[coin]['Date'])

    crypto[coin] = crypto[coin].sort_values('Date')

    crypto[coin] = crypto[coin].set_index(crypto[coin]['Date'])

    crypto[coin] = crypto[coin].drop('Date', 1)
for coin in crypto:

    print(coin, len(crypto[coin]))
del crypto['bitcoin_cash'], crypto['numeraire'], crypto['iota']
cryptoAll = {} # for later on



for coin in crypto:

    cryptoAll[coin] = crypto[coin]

    crypto[coin] = crypto[coin][-350:]
# Differencing

for coin in crypto:

    crypto[coin]['CloseDiff'] = crypto[coin]['Close'].diff().fillna(0)
for coin in crypto:

    plt.plot(crypto[coin]['CloseDiff'], label=coin)

plt.legend(loc=2)

plt.title('Daily Differenced Closing Prices')

plt.show()
# Percent Change

for coin in crypto:

    crypto[coin]['ClosePctChg'] = crypto[coin]['Close'].pct_change().fillna(0)

    
for coin in crypto:

    plt.plot(crypto[coin]['ClosePctChg'], label=coin)

plt.legend(loc=2)

plt.title('Daily Percent Change of Closing Price')

plt.show()
for coin in crypto:

    plt.plot(crypto[coin]['ClosePctChg'][-30:], label=coin)

plt.legend(loc=2)

plt.title('Daily Percent Change of Closing Price')

plt.show()
for coin in crypto:

    print('\n',coin)

    adf = adfuller(crypto[coin]['ClosePctChg'][1:])

    print(coin, 'ADF Statistic: %f' % adf[0])

    print(coin, 'p-value: %f' % adf[1])

    print(coin, 'Critical Values', adf[4]['1%'])

    print(adf)
for coin in crypto:

    print('\n',coin)

    adf = adfuller(crypto[coin]['CloseDiff'][1:])

    print(coin, 'ADF Statistic: %f' % adf[0])

    print(coin, 'p-value: %f' % adf[1])

    print(coin, 'Critical Values', adf[4]['1%'])

    print(adf)
for coin in crypto:

    model = LinearRegression()

    model.fit(np.arange(350).reshape(-1,1), crypto[coin]['Close'].values)

    trend = model.predict(np.arange(350).reshape(-1,1))

    plt.subplot(1, 2, 1)

    plt.plot(trend, label='trend')

    plt.plot(crypto[coin]['Close'].values)

    plt.title(coin)

    

    plt.subplot(1, 2, 2)

    plt.plot(crypto[coin]['Close'].values - trend, label='residuals')

    plt.title(coin)

    

    plt.show()
corrBitcoin = {}

corrDF = pd.DataFrame()



for coin in crypto: 

    corrBitcoin[coin] = correlate(crypto[coin]['ClosePctChg'], crypto['bitcoin']['ClosePctChg'])

    lag = np.argmax(corrBitcoin[coin])

    laggedCoin = np.roll(crypto[coin]['ClosePctChg'], shift=int(np.ceil(lag)))

    corrDF[coin] = laggedCoin

    

    plt.figure(figsize=(15,10))

    plt.plot(laggedCoin)

    plt.plot(crypto['bitcoin']['ClosePctChg'].values)

    title = coin + '/bitcoin PctChg lag: ' + str(lag-349)

    plt.title(title)



    plt.show()
font = {'family': 'serif',

        'color':  'black',

        'weight': 'normal',

        'size': 20,

        }



plt.matshow(corrDF.corr(method='pearson'))

plt.xticks(range(10), corrDF.columns.values, rotation='vertical')

plt.yticks(range(10), corrDF.columns.values)

plt.xlabel('Pearson Correlation', fontdict=font)

plt.show()

corrDF.corr(method='pearson')
plt.matshow(corrDF.corr(method='spearman'))

plt.xticks(range(10), corrDF.columns.values, rotation='vertical')

plt.yticks(range(10), corrDF.columns.values)

plt.xlabel('Spearman Correlation', fontdict=font)

plt.show()

corrDF.corr(method='spearman')
plt.matshow(corrDF.corr(method='kendall'))

plt.xticks(range(10), corrDF.columns.values, rotation='vertical')

plt.yticks(range(10), corrDF.columns.values)

plt.xlabel('Kendall Correlation', fontdict = font)

plt.show()

corrDF.corr(method='kendall')