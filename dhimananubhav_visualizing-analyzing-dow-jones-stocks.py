%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_context("notebook")



import warnings

warnings.filterwarnings("ignore")



import glob

filelist = glob.glob("../input/stock-data-dow-jones/*.csv")



# Load data

AAPL = pd.read_csv('../input/stock-data-dow-jones/AAPL.csv', sep=',', index_col=0, parse_dates=True)

cols = ['open', 'high', 'low', 'close', 'volume']

AAPL = AAPL[cols].copy()

AAPL = AAPL[AAPL.index>='2015-01-01']



df_stocks = pd.DataFrame()



for i in filelist:

    tmp = pd.read_csv(i)

    tmp['symbol'] = i.split('/')[-1].split('.')[0]

    df_stocks = df_stocks.append(tmp)



cols = ['date', 'close', 'volume', 'symbol']

df_stocks = df_stocks[cols].copy()

df_stocks['date'] = pd.to_datetime(df_stocks['date'])



# create new dataframe with just closing price for each stock

df = df_stocks.pivot(index='date', columns='symbol', values='close')



# subset data since 2015

df = df[df.index>='2015-01-01']

df.tail()
plt.figure(figsize=(15,7))

top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)

bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)

top.plot(AAPL.index, AAPL['close']) 

bottom.bar(AAPL.index, AAPL['volume']) 

 

# set the labels

top.axes.get_xaxis().set_visible(False)

top.set_title('AAPL')

top.set_ylabel('Closing Price')

bottom.set_ylabel('Volume');
plt.figure(figsize=(15,7))

sns.distplot(AAPL['close'].dropna(), bins=50, color='purple');
# simple moving averages

sma5 = AAPL['close'].rolling(5).mean() #5 days

sma100 = AAPL['close'].rolling(100).mean() #100 days

 

AAPL_sma = pd.DataFrame({'AAPL': AAPL['close'], 'SMA 5': sma5, 'SMA 100': sma100})

AAPL_sma.plot(figsize=(15, 7), legend=True, title='AAPL');
df.plot(figsize=(15,8))

plt.ylabel('Price');
returnfstart = df.apply(lambda x: x / x[0])

returnfstart.plot(figsize=(15,7)).axhline(1, lw=1, color='black')

plt.ylabel('Return From Start Price');
df.pct_change().plot(figsize=(15,7))

plt.axhline(0, color='black', lw=1)

plt.ylabel('Daily Percentage Return');
# Resample df to business months, take last observation as value 

monthly = df.resample('BM').apply(lambda x: x[-1])



# Calculate the monthly percentage change

monthly.pct_change().plot(figsize=(15,7))

plt.axhline(0, color='black', lw=1)

plt.ylabel('Monthly Percentage Return');
# Resample df to quarters, take the mean as value per quarter

quarter = df.resample('4M').mean()



# Calculate the quarterly percentage change

quarter.pct_change().plot(figsize=(15,7))

plt.axhline(0, color='black', lw=1)

plt.ylabel('Quarterly Percentage Return');
sns.jointplot('AAPL', 'NKE', df, kind='reg', color='seagreen');
# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
fig = sns.PairGrid(df[['AAPL', 'NKE', 'WBA']].dropna())

# define top, bottom and diagonal plots

fig.map_upper(plt.scatter, color='purple')

fig.map_lower(sns.kdeplot, cmap='cool_d')

fig.map_diag(sns.distplot, bins=30);
daily_pct_c = df/ df.shift(1) -1

daily_pct_c.hist(bins=50, figsize=(15,20))

plt.show()



display(daily_pct_c.describe())
cum_daily_return = (1 + daily_pct_c).cumprod()



cum_daily_return.plot(figsize=(12,8))

plt.ylabel('Cumulative Daily Returns');

plt.axhline(1, color='black', lw=1)

plt.show()
# Resample the cumulative daily return to cumulative monthly return 

cum_monthly_return = cum_daily_return.resample("M").mean()



cum_monthly_return.plot(figsize=(12,8))

plt.ylabel('Cumulative Monthly Returns');

plt.axhline(1, color='black', lw=1)

plt.show()
# Define the minumum of periods to consider 

min_periods = 75 



# Calculate the volatility

vol = daily_pct_c.rolling(min_periods).std() * np.sqrt(min_periods) 



# Plot the volatility

vol.plot(figsize=(10, 8))



# Show the plot

plt.show()
AAPL.head()
# Initialize the short and long windows

short_window = 40

long_window = 100



# Initialize the `signals` DataFrame with the `signal` column

signals = pd.DataFrame(index=AAPL.index)

signals['signal'] = 0.0



# Create short simple moving average over the short window

signals['short_mavg'] = AAPL['close'].rolling(window=short_window, min_periods=1, center=False).mean()



# Create long simple moving average over the long window

signals['long_mavg'] = AAPL['close'].rolling(window=long_window, min_periods=1, center=False).mean()



# Create signals

signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 

                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   



# Generate trading orders

signals['positions'] = signals['signal'].diff()
# Initialize the plot figure

fig = plt.figure(figsize=(10, 8))



# Add a subplot and label for y-axis

ax1 = fig.add_subplot(111,  ylabel='Price in $')



# Plot the closing price

AAPL['close'].plot(ax=ax1, color='grey', lw=2.)



# Plot the short and long moving averages

signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)



# Plot the buy signals

ax1.plot(signals.loc[signals.positions == 1.0].index, 

         signals.short_mavg[signals.positions == 1.0],

         '^', markersize=10, color='m')

         

# Plot the sell signals

ax1.plot(signals.loc[signals.positions == -1.0].index, 

         signals.short_mavg[signals.positions == -1.0],

         'v', markersize=10, color='k')

         

# Show the plot

plt.show()
# Set the initial capital

initial_capital= float(100000.0)



# Create a DataFrame `positions`

positions = pd.DataFrame(index=signals.index).fillna(0.0)



# Buy a 100 shares

positions['AAPL'] = 100*signals['signal']   

  

# Initialize the portfolio with value owned   

portfolio = positions.multiply(AAPL['close'], axis=0)



# Store the difference in shares owned 

pos_diff = positions.diff()



# Add `holdings` to portfolio

portfolio['holdings'] = (positions.multiply(AAPL['close'], axis=0)).sum(axis=1)



# Add `cash` to portfolio

portfolio['cash'] = initial_capital - (pos_diff.multiply(AAPL['close'], axis=0)).sum(axis=1).cumsum()   



# Add `total` to portfolio

portfolio['total'] = portfolio['cash'] + portfolio['holdings']



# Add `returns` to portfolio

portfolio['returns'] = portfolio['total'].pct_change()
fig = plt.figure(figsize=(10, 8))



ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')



# Plot the equity curve in dollars

portfolio['total'].plot(ax=ax1, lw=2.)



# Plot the "buy" trades against the equity curve

ax1.plot(portfolio.loc[signals.positions == 1.0].index, 

         portfolio.total[signals.positions == 1.0],

         '^', markersize=10, color='m')



# Plot the "sell" trades against the equity curve

ax1.plot(portfolio.loc[signals.positions == -1.0].index, 

         portfolio.total[signals.positions == -1.0],

         'v', markersize=10, color='k')



# Show the plot

plt.show()
# Isolate the returns of your strategy

returns = portfolio['returns']



# annualized Sharpe ratio

sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())



# Print the Sharpe ratio

print(np.round(sharpe_ratio,2))
# Define a trailing 252 trading day window

window = 252



# Calculate the max drawdown in the past window days for each day

rolling_max = AAPL['close'].rolling(window, min_periods=1).max()

daily_drawdown = AAPL['close']/rolling_max - 1.0



# Calculate the minimum (negative) daily drawdown

max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()



# Plot the results

daily_drawdown.plot()

max_daily_drawdown.plot()



# Show the plot

plt.show()
# Get the number of days in `aapl`

days = (AAPL.index[-1] - AAPL.index[0]).days



# Calculate the CAGR 

cagr = ((((AAPL['close'][-1]) / AAPL['close'][1])) ** (365.0/days)) - 1



# Print CAGR

print(np.round(cagr,3))
# Drop the columns

ph_df = AAPL.drop(['open', 'high', 'low','volume'], axis=1)

ph_df.reset_index(inplace=True)

ph_df.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)

ph_df['ds'] = pd.to_datetime(ph_df['ds'])

ph_df.head()
from fbprophet import Prophet

m = Prophet()

m.fit(ph_df)



# Create Future dates

future_prices = m.make_future_dataframe(periods=365)



# Predict Prices

forecast = m.predict(future_prices)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = m.plot(forecast)

ax1 = fig.add_subplot(111)

ax1.set_title("AAPL Stock Price Forecast", fontsize=16)

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Close Price", fontsize=12)

plt.show()
fig2 = m.plot_components(forecast)

plt.show()
# Monthly Data Predictions

m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)

future = m.make_future_dataframe(periods=12, freq='M')

fcst = m.predict(future)

fig = m.plot(fcst)

plt.title("Monthly Prediction \n 1 year time frame", fontsize=16)

plt.xlabel("Date", fontsize=12)

plt.ylabel("Close Price", fontsize=12)



plt.show()
fig = m.plot_components(fcst)

plt.show()