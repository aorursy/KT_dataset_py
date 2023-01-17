import numpy as np 

import pandas as pd

import datetime

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from itertools import repeat, chain



%matplotlib inline
balance = 0.03
plt.rcParams['figure.figsize'] = (20, 15)
def dateparse (time_in_secs):    

    return datetime.datetime.fromtimestamp(float(time_in_secs))



data = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv', parse_dates=True,

                   date_parser=dateparse, index_col=[0])
data = data[data.index > '2017-01-01']
# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so

# lets fill forwards those values...

data['Open'].fillna(method='ffill', inplace=True)

data['High'].fillna(method='ffill', inplace=True)

data['Low'].fillna(method='ffill', inplace=True)

data['Close'].fillna(method='ffill', inplace=True)



# First thing is to fix the data for bars/candles where there are no trades. 

# Volume/trades are a single event so fill na's with zeroes for relevant fields...

data['Volume_(BTC)'].fillna(value=0, inplace=True)

data['Volume_(Currency)'].fillna(value=0, inplace=True)

data['Weighted_Price'].fillna(data['Close'], inplace=True)
prices = data['Weighted_Price']
prices[::10].plot(figsize=(20, 7))
pl_buy_and_hold = (prices[-1] * balance - prices[0] * balance) - prices[0] * balance

pl_buy_and_hold
n_trades = 100



idx_buys = np.random.choice(len(data)-100, (n_trades, ))

idx_buys = np.sort(idx_buys)

idx_sells = idx_buys + 100
pl = 0



np.random.shuffle(idx_buys)

np.random.shuffle(idx_sells)



for ibuy, isell in zip(idx_buys, idx_sells):

    pl += (prices[isell] - prices[ibuy]) * balance

    

pl
((prices[idx_sells].sum() - prices[idx_buys].sum()) * balance)
sample_size = 24 * 60 * 1

sample_start = len(prices) - 5 * sample_size

prices_sample = prices.iloc[sample_start:sample_start+sample_size]

prices_sample.plot()
def calc_pl(short_window, long_window, reverse_position=True):

    assert short_window < long_window



    # Apply moving average

    ma_short = prices_sample.rolling(short_window).mean()

    ma_long = prices_sample.rolling(long_window).mean()



    # Detect changes in position

    sign_changes = np.sign((ma_short/ma_long)-1).diff()



    # Find index of buys and sells signals

    buys = np.where(sign_changes == 2)[0]

    sells = np.where(sign_changes == -2)[0]



    # Keep len of buys/sells equals

    min_len = min(len(buys), len(sells))

    if min_len == 0:

        return 0

    buys = buys[-min_len:]

    sells = sells[-min_len:]



    assert len(buys) == len(sells)

    

    if reverse_position:

        # Calculate PL for reverse positions every signal

        if sells[0] < buys[0]:

            # Stat selling

            pl = prices_sample[sells[0]] + 2 * prices_sample[sells[1:]].sum() - 2 * prices_sample[buys[:-1]].sum() - prices_sample[buys[-1]]

        else:

            # Stat buying

            pl = -prices_sample[buys[0]] + 2 * prices_sample[sells[:-1]].sum() - 2 * prices_sample[buys[1:]].sum() + prices_sample[sells[-1]]

        pl *= balance

    else:

        # PL without position reversal (will close position every pair or signals)

        pl = ((prices_sample[sells].sum() - prices_sample[buys].sum()) * balance)

        

    return pl
short_window = 60

long_window = 240





calc_pl(short_window, long_window)
# Interactive version of PL calculus with position reversal

# orders = sorted(chain(zip(buys, repeat('buy')), zip(sells, repeat('sell'))))



# pl = 0

# for i, (idx_price, side) in enumerate(orders):

#     # Invert position if not first or last

#     order_size = 1 if i == 0 else 2

#     order_price = prices_sample[idx_price] * balance * order_size

#     direction = 1 if side == 'sell' else -1

#     pl += order_price * direction

#     print('{:7.2f} {:4s} {} {:.2f}'.format(pl-order_price * direction / order_size, side, order_size, order_price / order_size))

    

# # Close last position

# pl -= order_price * direction / 2

        

# pl
# ax = prices_sample.plot(label='price', figsize=(18, 8))

# ax2 = ax.twinx()

# ma_short.plot(ax=ax, label='short')

# ma_long.plot(ax=ax, label='long')

# #sign_changes.plot(ax=ax2, color='gray', linestyle='--')

# (ma_short/ma_long).plot(ax=ax2, color='gray', linestyle='--')

# ax2.axhline(y=1, color='k')

# #ax2.set_ylim(-1, 1)

# ax.legend()
shorts = np.arange(5, 60, 2)

longs = np.arange(60, 480, 10)

shorts, longs
im = np.zeros((len(shorts), len(longs)))

for i, s in enumerate(tqdm(shorts)):

    for j, l in enumerate(longs):

        im[i, j] = calc_pl(s, l)
vrange = 50 # max(im.max(), -im.min())

c = plt.imshow(im, vmin=-vrange, vmax=vrange, cmap='seismic_r', interpolation='bicubic')

plt.colorbar(c, shrink=0.65)

plt.xticks(np.arange(len(longs)), labels=longs, rotation=90)

plt.yticks(np.arange(len(shorts)), labels=shorts)

plt.xlabel('long')

plt.ylabel('short');
sample_size = 60 * 24 * 7 * 2  # In minutes

vrange = 50 # max(im.max(), -im.min())



for sample_start in tqdm(range(0, len(prices), sample_size)):

    prices_sample = prices.iloc[sample_start:sample_start+sample_size]

    dt_stat, dt_end = prices.index[[sample_start, min(len(prices)-1, sample_start+sample_size)]]

    

    im = np.zeros((len(shorts), len(longs)))

    for i, s in enumerate(shorts):

        for j, l in enumerate(longs):

            im[i, j] = calc_pl(s, l)

    

    fig, ax = plt.subplots(2, 1, figsize=(20, 15))

    

    prices_sample.plot(ax=ax[0])

    

    c = ax[1].imshow(im, vmin=-vrange, vmax=vrange, cmap='seismic_r', interpolation='bicubic', aspect='auto')

    fig.colorbar(c, shrink=0.65)

    ax[1].set_xticks(np.arange(len(longs)))

    ax[1].set_xticklabels(longs, rotation=90)

    ax[1].set_yticks(np.arange(len(shorts)))

    ax[1].set_yticklabels(shorts)

    ax[1].set_xlabel('long')

    ax[1].set_ylabel('short')

    fig.tight_layout()

    fig.savefig('period_{}_to_{}.png'.format(dt_stat.date(), dt_end.date()))

    plt.close()