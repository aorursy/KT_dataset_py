# What would happen if we would go long when a 50 day moving average crosses above a 100 day moving average, and close when it crosses back down?

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('/kaggle/input/qqq-from-april-2019-april-2020/qqq.csv', index_col='Date', parse_dates=['Date'])
data['SMA50'] = data['High'].rolling(50).mean()
data['SMA100'] = data['Low'].rolling(100).mean()

# Set to 1 if SMA50 is above SMA100
data['Position'] = np.where(data['SMA50'] > data['SMA100'], 1, 0)

# Buy a day delayed, shift the column
data['Position'] = data['Position'].shift()

# Calculate the daily percent returns of strategy
data['StrategyPct'] = data['High'].pct_change(1) * data['Position']

# Calculate cumulative returns
data['Strategy'] = (data['StrategyPct'] + 1).cumprod()

# Calculate index cumulative returns
data['BuyHold'] = (data['High'].pct_change(1) + 1).cumprod()

# Plot the result
data[['Strategy', 'BuyHold']].plot()
                           
