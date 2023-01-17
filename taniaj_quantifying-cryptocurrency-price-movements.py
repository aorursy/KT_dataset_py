import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline  

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 }, 
        palette=sns.color_palette("Blues_r", 20))

import warnings
warnings.filterwarnings('ignore')
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
all_currencies_df = pd.read_csv('../input/all_currencies.csv', parse_dates=['Date'], date_parser=dateparse)
all_currencies_df.sample(5)
# Get rank by marketcap
last_date = max(all_currencies_df.Date)
last_date_currencies_df = all_currencies_df[all_currencies_df['Date'] == last_date]

last_date_currencies_df['rank'] = last_date_currencies_df['Market Cap'].rank(method='dense', ascending=False)

all_currencies_df = pd.merge(left=all_currencies_df,right=last_date_currencies_df[['Symbol', 'rank']], how='left', left_on='Symbol', right_on='Symbol')

# Initially we will just work with the top 50 cryptos with data starting from 2017 to speed things up
top50_currencies_df = all_currencies_df[all_currencies_df['rank'] <= 50]
#top50_currencies_df = top50_currencies_df[(top50_currencies_df['Date'] >= '2017-01-01')]
top50_pivot_df = top50_currencies_df.pivot(index='Date', columns='Symbol', values='Close')
top50_corr = top50_pivot_df.corr()

cmap = sns.diverging_palette(240, 10,sep=20, as_cmap=True)

plt.figure(figsize = (16,16))
plt.tight_layout()
sns.heatmap(top50_corr, 
            xticklabels=top50_corr.columns.values,
            yticklabels=top50_corr.columns.values, 
            cmap=cmap, vmin=-1, vmax=1, annot=False, square=True)
latest_date = max(top50_currencies_df['Date'])
top50_currencies_latest_df = top50_currencies_df[top50_currencies_df['Date'] == latest_date]
corr_outliers_df = top50_currencies_latest_df[top50_currencies_latest_df['Symbol'].isin(['BCD', 'BTG', 'CNX', 'HSR', 'POA'])]
corr_outliers_df
top50_start_dates = top50_currencies_df.loc[top50_currencies_df.groupby('Symbol')['Date'].idxmin()]
corr_outliers_df = top50_start_dates[top50_start_dates['Symbol'].isin(['BCD', 'BTG', 'CNX', 'HSR', 'POA'])]
corr_outliers_df
top20_currencies_df = all_currencies_df[all_currencies_df['rank'] <= 20]

top20_pivot_df = top20_currencies_df.pivot(index='Date', columns='Symbol', values='Close')
top20_corr = top20_pivot_df.corr()

cmap = sns.diverging_palette(240, 10,sep=20, as_cmap=True)

plt.figure(figsize = (16,16))
plt.tight_layout()
sns.heatmap(top20_corr, 
            xticklabels=top20_corr.columns.values,
            yticklabels=top20_corr.columns.values, 
            cmap=cmap, vmin=0, vmax=1, annot=True, square=True)
# Reshape the dataframe to multiindex by Date and currency
top50_currencies_df.set_index(['Symbol', 'Date'], inplace=True) # Or should it be the other way around?
top50_currencies_df.head()
# Get daily percent change in close price
top50_currencies_df['pct_change'] = top50_currencies_df['Close'].groupby(level=0).pct_change()
top50_currencies_df.head()
# Plot daily percentage change for BTC
btc_df = top50_currencies_df.xs('BTC')
btc_df.head()
btc_df['pct_change'].plot(figsize=(16,8), title='BTC Daily Percentage Change')
plt.savefig('bitcoin_daily_pct_change.png')
# calculate mean price over the past 30 days
top50_currencies_df['ma_30'] = top50_currencies_df['pct_change'].rolling(window=30).mean()

top50_currencies_df['deviation'] = top50_currencies_df['pct_change'] - top50_currencies_df['ma_30']

top50_currencies_df['variance'] = top50_currencies_df['deviation']**2 / 30

top50_currencies_df.sample(5)
top50_average_variance_df = top50_currencies_df['variance'].groupby('Date').mean()
top50_average_variance_df.plot(figsize=(16,8), title='Daily Average Variance of Top50 Currencies')
# Lets look at just the past year
top50_average_variance_df = top50_average_variance_df.loc['2017-04-01':]
top50_average_variance_df.plot(figsize=(16,8), title='Daily Average Variance of Top50 Currencies from 2017-04-01')
top20_currencies_df = all_currencies_df[all_currencies_df['rank'] <= 20]
top20_start_dates = top20_currencies_df.loc[top20_currencies_df.groupby('Symbol')['Date'].idxmin()]
start_date = max(top20_start_dates.Date)
# Lets start this index at the earliest date when the top 20 cryptos all existed
top20_start_dates = top20_currencies_df.loc[top20_currencies_df.groupby('Symbol')['Date'].idxmin()]
# Determine index start date
start_date = max(top20_start_dates['Date'])
start_date
# Remove all older data
index_currencies_df = top20_currencies_df[(top20_currencies_df['Date'] >= start_date)]
index_currencies_df.sample()
# Detemine index divisor at the start date 
crypto20_index_divisor = index_currencies_df[index_currencies_df['Date']==start_date]['Market Cap'].sum() / 100
crypto20_index_divisor
# Build the index
def calculate_index_value(df_sub):
    return df_sub['Market Cap'].sum() / crypto20_index_divisor

crypto20_index_df = index_currencies_df.groupby('Date').apply(calculate_index_value)
crypto20_index_df.head()
crypto20_index_df.plot(figsize=(16,8), title='Crypto20 Index')
# Lets see just how much influence BTC has on this index
top20_latest_df = top20_currencies_df[all_currencies_df['Date'] == last_date]
top20_latest_df.sort_values(by='Market Cap',ascending=False).plot(x='Symbol', y='Market Cap', kind='bar', figsize=(16,8))
# Detemine initial amount of each currency assuming an index starting value of 100
n = 20
index_currencies_start_df = index_currencies_df[index_currencies_df['Date']==start_date]
index_currencies_start_df['eq_index_holdings'] = 100 / n / index_currencies_start_df['Close']
# Add eq_index_holdings column to index_currencies_df
index_currencies_df = pd.merge(left=index_currencies_df,right=index_currencies_start_df[['Symbol', 'eq_index_holdings']], how='left', left_on='Symbol', right_on='Symbol')
# Calculate the value for each date/Currency
index_currencies_df['eq_index_holdings_value'] = index_currencies_df['eq_index_holdings']*index_currencies_df['Close']
# Build index
crypto20eq_index_df = index_currencies_df.groupby('Date')['eq_index_holdings_value'].sum()
crypto20eq_index_df.head()
with sns.color_palette("Blues", 2):
    ax = crypto20_index_df.plot(figsize=(16,8), title='Crypto20 Index', label='Crypto20 Market Cap Index', legend=True)
    crypto20eq_index_df.plot(ax=ax, label='Crypto20 Equal Weight Index', legend=True)
crypto20eq_index_df = crypto20eq_index_df.reset_index()
crypto20eq_index_df.head()

# Crypto - We already have top50_currencies_df['pct_change'] from a previous step

# calculate the market variance based on the Crypto20 Equal Weight Index
crypto20eq_index_df['pct_change'] = crypto20eq_index_df['eq_index_holdings_value'].pct_change()

crypto20eq_index_df['ma_30'] = crypto20eq_index_df['pct_change'].rolling(window=30).mean()

crypto20eq_index_df['deviation'] = crypto20eq_index_df['pct_change'] - crypto20eq_index_df['ma_30']

crypto20eq_index_df['variance'] = crypto20eq_index_df['deviation']**2 / 30
crypto20eq_index_df.tail()
# calculate the covariance between the return of the crypto and the market return

#better way to join than resetting index?
top50_currencies_df = top50_currencies_df.reset_index()

# covariance
# This is probably not the most efficient way of achieving this calculation. If any python experts out there know a better way please let me know. In principle rolling windows across multiple dataframes?
top50_currencies_df =  pd.merge(left=top50_currencies_df,right=crypto20eq_index_df[['Date', 'pct_change', 'ma_30', 'deviation']], how='left', left_on='Date', right_on='Date')
top50_currencies_df['product'] = top50_currencies_df['deviation_x']*top50_currencies_df['deviation_y']


top50_currencies_df.tail()
top50_currencies_df['covariance'] = top50_currencies_df['product'].rolling(window=30).sum()/29
top50_currencies_df.tail()
# Plot the covariance
# Split into top10, middle20, bottom20 because the plots become too cluttered to read otherwise.
top_currencies_df = top50_currencies_df[top50_currencies_df['rank'] <= 10]
mid_currencies_df = top50_currencies_df[(top50_currencies_df['rank'] > 10) & (top50_currencies_df['rank'] <= 30)]
bottom_currencies_df = top50_currencies_df[top50_currencies_df['rank'] > 30]

top_currencies_cov_df = top_currencies_df.pivot(index='Date', columns='Symbol', values='covariance')

with sns.color_palette("husl", 10):
    top_currencies_cov_df.plot(figsize=(16,8), title='Covariance')
mid_currencies_cov_df = mid_currencies_df.pivot(index='Date', columns='Symbol', values='covariance')

with sns.color_palette("husl", 20):
    mid_currencies_cov_df.plot(figsize=(16,10), title='Covariance')
bottom_currencies_cov_df = bottom_currencies_df.pivot(index='Date', columns='Symbol', values='covariance')

with sns.color_palette("husl", 20):
    bottom_currencies_cov_df.plot(figsize=(16,10), title='Covariance')