import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.finance as mpl_finance

%matplotlib inline
bitcoin_df = pd.read_csv('../input/CHART_DATA_BITCOIN_2017.csv')

altcoin_df = pd.read_csv('../input/CHART_DATA_ALTCOINS_2017.csv')
print('The open price of 2017 was {:,.2f} USD.'.format(bitcoin_df.open[0]))

print('The close price of 2017 was {:,.2f} USD.'.format(bitcoin_df.close[364]))

print('The highest price of 2017 was {:,.2f} USD.'.format(max(bitcoin_df.high)))

print('Increase rate (close price / open price) was {:,.2f}.'.format(bitcoin_df.close[364] / bitcoin_df.open[0]))

print('Increase rate (highest price / open price) was {:,.2f}.'.format(max(bitcoin_df.high) / bitcoin_df.open[0]))
fig = plt.figure(figsize=(10, 5),dpi=200)

ax = plt.subplot()



ohlc = mpl_finance.candlestick2_ohlc(

    ax, bitcoin_df.open, bitcoin_df.high, bitcoin_df.low, bitcoin_df.close, width=1, colorup='k', colordown='r', alpha=0.5)



ax.grid()

ax.set_xlim(0, 365)

fig.autofmt_xdate()



plt.title('Bitcoin in 2017')

plt.xlabel('Day in 2017')

plt.ylabel('Price (USD)')
bitcoin_df.open[0], bitcoin_df.close[364], bitcoin_df.close[364] / bitcoin_df.open[0], max(bitcoin_df.high)
altcoin_transition_df = pd.merge(

    altcoin_df[altcoin_df['date'] == '2017-01-01'][['ticker', 'open']],

    altcoin_df[altcoin_df['date'] == '2017-12-31'][['ticker', 'close']],

    on='ticker')

altcoin_transition_df.index = altcoin_transition_df['ticker']
# Increase rate of each altcoins

altcoin_transition_df['increase_rate'] = altcoin_transition_df.close / altcoin_transition_df.open
# Transaction volume of each altcoins

altcoin_transition_df = altcoin_transition_df.join(altcoin_df.groupby('ticker')[['volume']].mean())

altcoin_transition_df['average_volume'] = altcoin_transition_df['volume']
altcoin_transition_df.index = altcoin_transition_df['ticker'].apply(lambda x: x.replace('BTC_', ''))
fig = plt.figure(figsize=(8, 8),dpi=200)

ax = plt.subplot()

altcoin_transition_df.plot(kind='scatter', x='average_volume', y='increase_rate', loglog=True, ax=ax)



ax.axhline(y = 1, color = 'grey', alpha = 0.75)



for k, v in altcoin_transition_df.iterrows():

    ax.annotate(k, xy=(v.average_volume, v.increase_rate), size=8)



plt.title('Altcoins in 2017')

plt.xlabel('Average transaction volume (BTC)')

plt.ylabel('Altcoins increase rate comparing Bitcoin')
altcoins_count = len(altcoin_transition_df['increase_rate'])

better_than_bitcoin_count = sum(altcoin_transition_df['increase_rate'] > 1)

print('The altcoins count for survey is {}.'.format(altcoins_count))

print('{} altcoins ({:.2f}%) were better than Bitcoin in 2017.'.format(better_than_bitcoin_count, better_than_bitcoin_count / altcoins_count * 100))