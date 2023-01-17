import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
df = pd.read_csv('../input/kiva_loans.csv')
df.head()
df.info()
pd.isnull(df).sum()
df.funded_time = df['funded_time'].fillna(df['posted_time'])
pd.isnull(df).sum()
df.posted_time = np.array(df.posted_time,dtype='datetime64[D]')
df.disbursed_time = np.array(df.disbursed_time,dtype='datetime64[D]')
df.funded_time = np.array(df.funded_time,dtype='datetime64[D]')
df.date = np.array(df.date,dtype='datetime64[D]')
df.info()
df.head()
df_sub = df[['funded_amount','date']]
df_sub = df_sub.set_index('date')
plt.style.use('fivethirtyeight')

ax = df_sub.plot(linewidth=1,figsize=(20, 6), fontsize=10)
ax.set_xlabel('Date')
plt.show()
df_sub_1 = df_sub['2014-01-01':'2014-12-31']

ax = df_sub_1.plot(linewidth=1,figsize=(20, 6), fontsize=10)
ax.set_xlabel('Date')

ax.axvline('2014-06-10', color='red', linestyle='-',linewidth=3,alpha = 0.3)

plt.show()
mov_avg = df_sub.rolling(window=52).mean()

mstd = df_sub.rolling(window=52).std()

mov_avg['upper'] = mov_avg['funded_amount'] + (2 * mstd['funded_amount'])
mov_avg['lower'] = mov_avg['funded_amount'] - (2 * mstd['funded_amount'])

ax = mov_avg.plot(linewidth=0.8,figsize=(20, 6) , fontsize=10,alpha = 0.5)
ax.set_title('Rolling mean and variance of Fund \n from 2013 to 2017', fontsize=10)
index = df_sub.index.week

df_sub_week = df_sub.groupby(index).mean()

mov_avg = df_sub_week.rolling(window=4).mean()

mstd = df_sub_week.rolling(window=4).std()

mov_avg['upper'] = mov_avg['funded_amount'] + (2 * mstd['funded_amount'])
mov_avg['lower'] = mov_avg['funded_amount'] - (2 * mstd['funded_amount'])

ax = mov_avg.plot(linewidth=0.8,figsize=(20, 6) , fontsize=10)
ax.set_title('Rolling mean and variance of Fund \n from 2013 to 2017 in weeks', fontsize=10)
ax = plt.subplot()

ax.boxplot(df_sub['funded_amount'])
ax.set_yscale('log')
ax.set_xlabel('fund')
ax.set_title('Distribution of funds', fontsize=10)
plt.show()
df_sub.describe()
ax = df_sub.plot(kind='density',figsize=(20, 6) , linewidth=3, fontsize=10)
ax.set_xlabel('fund')
ax.set_ylabel('density')
ax.set_xscale('log')
ax.set_title('Density of funds', fontsize=10)
plt.show()
fig = tsaplots.plot_acf(df_sub_week['funded_amount'], lags=24)
plt.show()
df_sub_mean = df_sub.groupby(pd.TimeGrouper('D')).mean().dropna()
df_sub_total = df_sub.groupby(pd.TimeGrouper('D')).sum().dropna()
decomposition = sm.tsa.seasonal_decompose(df_sub_total)

trend = decomposition.trend

ax = trend.plot(figsize=(20, 6), fontsize=6)

ax.set_xlabel('Date', fontsize=10)
ax.set_title('Seasonal component of total fund', fontsize=10)
plt.show()
df['date'] = pd.to_datetime(df['date'])

df = df.set_index('date')
new_df = df[['funded_amount','lender_count']]
new_df_mean = new_df.groupby(pd.TimeGrouper('D')).mean().dropna()
new_df_total = new_df.groupby(pd.TimeGrouper('D')).sum().dropna()

new_df_mean.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()
new_df_total.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()
new_df_mean = new_df.groupby(pd.TimeGrouper('W')).mean().dropna()
new_df_total = new_df.groupby(pd.TimeGrouper('W')).sum().dropna()

new_df_mean.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()
new_df_total.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()
new_df_mean = new_df.groupby(pd.TimeGrouper('M')).mean().dropna()
new_df_total = new_df.groupby(pd.TimeGrouper('M')).sum().dropna()

new_df_mean.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()
new_df_total.plot(subplots=True, 
          layout=(2,1), 
          sharex=True, 
          sharey=False, 
          figsize=(20, 6),
          fontsize=8, 
          legend=True,
          linewidth=0.5)

plt.show()
df['Time_Spent'] = df['posted_time'] - df['disbursed_time']
df.head()
type(df.Time_Spent)

