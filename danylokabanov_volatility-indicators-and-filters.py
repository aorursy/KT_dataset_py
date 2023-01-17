import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


origin_df = pd.read_csv('/kaggle/input/392-crypto-currency-pairs-at-minute-resolution/btcusd.csv')
origin_df.set_index('time')
origin_df['close_mean']=  origin_df['close'].rolling(14).mean()
origin_df['close_std']=  origin_df['close'].rolling(14).std()
# Get data for January 2017
df = origin_df
df['time'] = pd.to_datetime(df['time'], unit='ms')
df = df.loc[((df['time'] >= '2017-01-01 00:00:00') & (df['time'] < '2017-02-01 00:00:00'))]
df.reset_index(drop=True, inplace=True)

# Illustrate the close price
fig, ax = plt.subplots()

ax.plot(df['time'], df['close'])
ax.set_title("Bitcoin to USD close price for January 2017 (Bitfinex)")
fig.autofmt_xdate()
ax.set_ylabel("Close price")

fig.set_dpi(150)
fig.show()
# Illustrate true range indicator

# 1. Get distance from High and Low
# 2. Get distance from High and Close
# 3. Get distance from Low and Close
# Result: Greatest from above is a True Range

df['tr1'] = (df['high'] - df['low']).abs()
df['tr2'] = (df['high'] - df['close']).abs()
df['tr3'] = (df['low'] - df['close']).abs()

df['true_range'] = df.apply(lambda row: max(row['tr1'], row['tr2'], row['tr3']), axis=1)

fig, ax = plt.subplots()
ax.plot(df['time'], df['true_range'])
ax.set_ylabel("True Range Indicator")
ax.set_title("Bitcoin to USD True Range Indicator for January 2017 (Bitfinex)")

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()
# Average True Range (14 days lookback period)

df['avg_true_range'] =  df['true_range'].rolling(14).mean()

fig, ax = plt.subplots()
ax.plot(df['time'], df['avg_true_range'])
ax.set_ylabel("Average True Range Indicator")
ax.set_title("Bitcoin to USD Average True Range Indicator for January 2017 (Bitfinex)")

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()
# Relative Strength Index (14 days lookback period)

window = 14

# 1. Find close differences (price up and down)
df['close_diff'] = df['close'] - df.shift()['close']
df['close_up'] = df.apply(lambda row: max(row['close_diff'], 0), axis=1)
df['close_down'] = df.apply(lambda row: abs(min(row['close_diff'], 0)), axis=1)
df = df.drop(columns=['close_diff'])


# 2. Find averages for aboves
df['avg_close_up'] = df['close_up'].rolling(window).mean()
df['avg_close_down'] = df['close_down'].rolling(window).mean()


# 3. Calculate Relative Strengh
df['rs_close'] = df['avg_close_up'] / df['avg_close_down']


# 4. Calculate Relative Strength Index
df['rsi_close'] = 100 - (100 / (1 + df['rs_close']))

# 5. Plot the results
fig, ax = plt.subplots()
ax.plot(df['time'], df['rsi_close'])
ax.set_ylabel("Relative Strength Index")
ax.set_title("Bitcoin to USD Relative Strength Index for January 2017 (Bitfinex)")
ax.grid(True)

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()


# Adjusted RSI (volume)

# 1. Find Relative Strengh for volume

# a) Find close differences (price up and down)
df['volume_diff'] = df['volume'] - df.shift()['volume']
df['volume_up'] = df.apply(lambda row: max(row['volume_diff'], 0), axis=1)
df['volume_down'] = df.apply(lambda row: abs(min(row['volume_diff'], 0)), axis=1)
df = df.drop(columns=['volume_diff'])

# 2. Find averages for aboves
df['avg_volume_up'] = df['volume_up'].rolling(window).mean()
df['avg_volume_down'] = df['volume_down'].rolling(window).mean()

# 3. Calculate Relative Strengh
df['rs_volume'] = df['avg_volume_up'] / df['avg_volume_down']

# 4. Find adjusted RSI
df['rsi_adjusted_close'] = 50 * (1 + (1 / (1 + df['rs_volume'])) - (1 / (1 + df['rs_close'])))

# 5. Plot the results
fig, ax = plt.subplots()
ax.plot(df['time'], df['rsi_adjusted_close'])
ax.set_ylabel("Relative Strength Index (adsjuted)")
ax.set_title("Bitcoin to USD Relative Strength Index (adjusted) for January 2017 (Bitfinex)")
ax.grid(True)

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()

# RSI (EMA)
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
# Here I don't use custom implementation for EMA because of Python limitations for recursive functions
# import sys
# sys.setrecursionlimit(300000)

window = 14
alpha = 2 / (window + 1)

# Custom EMA implementation
def ema_close(row):
    if row.name == 0:
        previous_ema = row['close_up']
    else:
        previous_ema = ema_close(row.shift()['close_up'])
        
    return alpha * row['close_up'] + (1 - alpha) * previous_ema
#df['close_up_ema'] = df.apply(ema_close, axis=1)



# 1. Find RS for close and volume based on EMA
# close
df['close_up_ema'] = df['close_up'].ewm(alpha=alpha, adjust=False).mean()
df['close_down_ema'] = df['close_down'].ewm(alpha=alpha, adjust=False).mean()
df['rs_close_ema'] = df['close_up_ema'] / df['close_down_ema']
# volume
df['volume_up_ema'] = df['volume_up'].ewm(alpha=alpha, adjust=False).mean()
df['volume_down_ema'] = df['volume_down'].ewm(alpha=alpha, adjust=False).mean()
df['rs_volume_ema'] = df['volume_up_ema'] / df['volume_down_ema']

# 2. Find adjusted RSI
df['rsi_adjusted_ema_close'] = 50 * (1 + (1 / (1 + df['rs_volume_ema'])) - (1 / (1 + df['rs_close_ema'])))

# 3. Plot the results
fig, ax = plt.subplots()
ax.plot(df['time'], df['rsi_adjusted_ema_close'])
ax.set_ylabel("Relative Strength Index (adsjuted & EMA)")
ax.set_title("Bitcoin to USD Relative Strength Index (adjusted & EMA) for January 2017 (Bitfinex)")
ax.grid(True)

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()
# MACD


# 1. Find 12 days EMA
window = 12
alpha = 2 / (window + 1)
df['close_ema_12'] = df['close'].ewm(alpha=alpha, adjust=False).mean()

# 2. Find 26 days EMA
window = 26
alpha = 2 / (window + 1)
df['close_ema_26'] = df['close'].ewm(alpha=alpha, adjust=False).mean()

# 3. Count the difference
df['close_MACD'] = df['close_ema_26'] - df['close_ema_12']

# 4. Plot the results
fig, ax = plt.subplots()
ax.plot(df['time'], df['close_MACD'])
ax.set_ylabel("MACD for close price")
ax.set_title("Bitcoin to USD MACD for January 2017 (Bitfinex)")
ax.grid(True)

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()
# Volatility Index (14 days)

window = 14

# 1. Calculate index
df['volatility_index'] = df['true_range']
df['volatility_index'] = (df['true_range'] + df.shift()['volatility_index'] * (window - 1)) / window

# 2. Plot the results
fig, ax = plt.subplots()
ax.plot(df['time'], df['volatility_index'])
ax.set_ylabel("Volatility Index for close price")
ax.set_title("Bitcoin to USD Volatility Index for January 2017 (Bitfinex)")
ax.grid(True)

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()
# Bollinger Bands Index

# 1. Find BB
df['close_bb_index'] = (4 * df['close_std']) / df['close_mean']

# 2. Plot the results
fig, ax = plt.subplots()
ax.plot(df['time'], df['close_bb_index'])
ax.set_ylabel("BB Index for close price")
ax.set_title("Bitcoin to USD BB Index for January 2017 (Bitfinex)")
ax.grid(True)

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()
# Donchian Channel
window = 14
def dc_channel(row):
    return (row.max() + row.min()) / 2

df['close_dc_channel'] = df.rolling(window)['close'].apply(dc_channel, raw=True)


# 2. Plot the results
fig, ax = plt.subplots()

ax.plot(df['time'][:100], df['close_dc_channel'][:100], '-b', label = "dc")
ax.plot(df['time'][:100], df['close'][:100], '-', label = "origin")

ax.set_ylabel("Close price")
ax.set_title("Bitcoin to USD Donchian Channel for 01.01.2017 (00:00 - 03:30) (Bitfinex)")

ax.grid(True)
ax.legend()

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()
# Keltner Channel
window = 14
alpha = 2 / (window + 1)

# 1. Upper KC = EMA(n) + 2*ATR(n)
df['close_upper_kc'] = df['close'].ewm(alpha=alpha, adjust=False).mean() + 2 * df['avg_true_range']
# 2. Lower KC = EMA(n) - 2*ATR(n)
df['close_lower_kc'] = df['close'].ewm(alpha=alpha, adjust=False).mean() - 2 * df['avg_true_range']

# 3. Index = (Upper + Lower) / EMA = 4 * ATR / EMA
df['close_kc_index'] = (df['close_upper_kc'] + df['close_lower_kc']) / 2


# 2. Plot the results
fig, ax = plt.subplots()

ax.plot(df['time'][:100], df['close_upper_kc'][:100], '-r', label = "kc upper")
ax.plot(df['time'][:100], df['close_lower_kc'][:100], '-b', label = "kc lower")
ax.plot(df['time'][:100], df['close_kc_index'][:100], '-g', label = "kc index")
ax.plot(df['time'][:100], df['close'][:100], '-', label = "origin")

ax.set_ylabel("Close price")
ax.set_title("Bitcoin to USD Keltner Channels for 01.01.2017 (00:00 - 03:30) (Bitfinex)")

ax.grid(True)
ax.legend()

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()


# Linear Regression Indicator
# y = a + b * x, where y - price, x - current timestamp
# a = (sum(y) - b * sum(x)) / n
# b = (n*sum(x * y) - sum(x) * sum(y)) / ( n*sum(x**2) - sum(x)**2 )

start = 0
n = 100
linear_df = df[start:start + n]
linear_df['timestamp'] = linear_df['time'].values.astype(np.int64) // 10 ** 9

def alpha_beta(x, y):
    beta = ( n * (x * y).sum() - x.sum() * y.sum() ) / ( n * (x**2).sum() - x.sum()**2)
    alpha = (y.sum() - beta * x.sum()) / n
    return alpha, beta


# MID
alpha, beta = alpha_beta(linear_df['timestamp'], linear_df['close'])
linear_df['linear_index'] = alpha + beta * linear_df['timestamp']

# UPPER
alpha, beta = alpha_beta(linear_df['timestamp'], linear_df['close'] + 3*df['close_std'])
linear_df['linear_index_upper'] = alpha + beta * linear_df['timestamp']

# LOWER
alpha, beta = alpha_beta(linear_df['timestamp'], linear_df['close'] - 3*df['close_std'])
linear_df['linear_index_lower'] = alpha + beta * linear_df['timestamp']

fig, ax = plt.subplots()
ax.plot(linear_df['time'], linear_df['linear_index_upper'], '-r', label = "linear up (3 SD)")
ax.plot(linear_df['time'], linear_df['linear_index'], '-g', label = "linear")
ax.plot(linear_df['time'], linear_df['linear_index_lower'], '-b', label = "linear low (3 SD)")
ax.plot(linear_df['time'], linear_df['close'], '-', label = "origin")

ax.set_ylabel("Close for close price")
ax.set_title("Bitcoin to USD Linear Regression for 01.01.2017 (00:00 - 03:30) (Bitfinex)")

ax.grid(True)
ax.legend()

fig.set_dpi(150)
fig.autofmt_xdate()
fig.show()