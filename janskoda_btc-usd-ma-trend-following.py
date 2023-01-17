import numpy as np

import pandas as pd
# Load BTC/USD minute candles

df = pd.read_csv('/kaggle/input/392-crypto-currency-pairs-at-minute-resolution/cryptominuteresolution/btcusd.csv')

df['time'] = pd.to_datetime(df['time'], unit = 'ms')

# Use data from 2015 until the end of dataset, that is Sep 2019

df = df[

    (df.time > pd.Timestamp(year = 2015, month = 1, day = 1)) &

    (df.time < pd.Timestamp(year = 2019, month = 1, day = 1))

].reset_index(drop = True)

# Drop the columns we don't use

df.drop(columns = ['high', 'low', 'volume'], inplace = True)

df
# Trading parameters; windows in minutes

win1 = 30 * 60

win2 = 1 * 60

# 0.01 = 1% price shift within the window

threshold1 = 0.009 

threshold2 = 0.005



# win1 = 300

# win2 = 3600

# threshold1 = 0.01

# threshold2 = -0.005



# Features & Trading logic

df['ma1'] = df['close'].rolling(win1).mean()

df['ma2'] = df['close'].rolling(win2).mean()

df['position'] = np.nan

# When price is above long-term MA and under short-term MA, BUY. 

# And vice versa. Keep +-1 positions, never close.

df.loc[df[

    (df.close > df.ma1 + threshold1 * df.close) & 

    (df.close < df.ma2 - threshold2 * df.close)

].index,'position'] = 1

df.loc[df[

    (df.close < df.ma1 - threshold1 * df.close) & 

    (df.close > df.ma2 + threshold2 * df.close)

].index,'position'] = -1



# Backtest

fees = 0.0010 # exchange transaction fee

df.loc[0, 'position'] = 0

df['position'].fillna(method = 'ffill', inplace = True)

df['nominal'] = - df['position'].diff() * df['close']

df['fees'] = df.position.diff().abs() * fees * df['close']

df['nominal'] -= df['fees']

df['earns'] = df['nominal'].cumsum() + df['position'] * df['close']

earns = df.earns.iloc[-1]



our_trades = df[df.position.diff() != 0].copy()

our_trades['duration'] = our_trades.time.diff()

our_trades['trade_earns'] = our_trades.earns.diff()

print(f'Earns = ${earns:.0f}')

print(f'Trades = {len(our_trades)}')

print(f'Profit per trade = ${earns / len(our_trades):.1f}')

df[['earns','close']].plot()
# Generate random walk data

np.random.seed(9)

df['noise'] = (np.random.rand(len(df)) - 0.5)

df['close'] = 1000 + df['noise'].cumsum()

df['open'] = np.nan

df
best_earns = 0.0

best_parameters = None



for win1 in [30,60,300,1800,3600,7200]:

   for win2 in [30,60,300,1800,3600,7200]:

       for threshold1 in [0.001, 0.005, 0.01, 0.02]:

            for threshold2 in [0.001, 0.005, -0.001, -0.005]:

                # Features & Trading logic

                df['ma1'] = df['close'].rolling(win1).mean()

                df['ma2'] = df['close'].rolling(win2).mean()

                df['position'] = np.nan

                # When price is above long-term MA and under short-term MA, BUY. And vice versa. Keep +-1 positions, never close.

                df.loc[df[(df.close > df.ma1 + threshold1 * df.close) & (df.close < df.ma2 - threshold2 * df.close)].index,'position'] = 1

                df.loc[df[(df.close < df.ma1 - threshold1 * df.close) & (df.close > df.ma2 + threshold2 * df.close)].index,'position'] = -1



                # Backtest

                fees = 0.0010 # exchange transaction fee

                df.loc[0, 'position'] = 0

                df['position'].fillna(method = 'ffill', inplace = True)

                df['nominal'] = - df['position'].diff() * df['close']

                df['fees'] = df.position.diff().abs() * fees * df['close']

                df['nominal'] -= df['fees']

                df['earns'] = df['nominal'].cumsum() + df['position'] * df['close']

                earns = df.earns.iloc[-1]



                if earns <= 0 or earns == np.nan: 

                    continue

                if earns > best_earns:

                    best_earns = earns

                    best_parameters = (win1, win2, threshold1, threshold2)

                print(f'{win1:5d} {win2:5d} {threshold1:.5f} {threshold2:.5f} {earns:+7.0f}')

                

print('Best earns', best_earns)

print('Best parameters', best_parameters)
# our_trades[our_trades.duration > pd.Timedelta(days = 10)]

# our_trades[our_trades.trade_earns.abs() > 20]