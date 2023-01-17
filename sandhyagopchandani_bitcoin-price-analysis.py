# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

from datetime import datetime

import seaborn as sns

color = sns.color_palette()

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import sys

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
bit_price_df = pd.read_csv('../input/bitcoin_price.csv')

for i, row in bit_price_df.iterrows():

    dt = datetime.strptime(row['Date'], '%b %d, %Y')        

    dt = dt.strftime('%Y-%m-%d')

    row['Date'] = dt

    bit_price_df.set_value(i,'Date',dt)
bit_price_df.head()
bitcoin_dataset_df = pd.read_csv('../input/bitcoin_dataset.csv')



for i, row in bitcoin_dataset_df.iterrows():

    dt = datetime.strptime(row['Date'], '%Y-%m-%d 00:00:00')        

    dt = dt.strftime('%Y-%m-%d')

    row['Date'] = dt

    bitcoin_dataset_df.set_value(i,'Date',dt)
bitcoin_dataset_df.head()
joined_data = bitcoin_dataset_df.merge(bit_price_df, on='Date')
joined_data.head()
sns.distplot(joined_data['Close'], kde=False, label='closing price') #default bins using Freedman-Diaconis rule.

#sns.distplot(joined_data['Open'], kde=False, label='Open price') #default bins using Freedman-Diaconis rule.

#sns.distplot(joined_data['High'], kde=False, label='High price') #default bins using Freedman-Diaconis rule.

plt.title("Distribution of closing price of Bitcoin")

plt.legend(loc='best')

plt.show()
import datetime



fig, ax = plt.subplots(figsize=(12,8))

x3 = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in joined_data.Date]



#joined_data

joined_data['moving_avg'] =  joined_data['Close'].rolling(window=30).mean()

#print(joined_data)





plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.plot(x3, joined_data.Close, label='closing price')

plt.plot(x3, joined_data.moving_avg, color='red', label='moving average(close price)')

plt.gcf().autofmt_xdate()

plt.xlabel("Date", fontsize=15)

plt.ylabel("Bitcoin Price", fontsize=15)

plt.title("Bitcoin Price overtime", fontsize=20)

plt.legend(loc='best')

plt.show()
joined_data['weekday'] = pd.to_datetime(joined_data['Date']).dt.weekday_name

week_data = joined_data.groupby(['weekday'], as_index=False)['Close'].agg({'mean': 'mean'})

day_of_week = pd.DataFrame(data=week_data)





plt.plot(figsize=(12,8))

plt.title('Day of Week Analysis')



my_xticks = np.array(day_of_week.weekday)

plt.xticks(range(len(week_data['mean'])), my_xticks)

plt.plot(range(len(week_data['mean'])), week_data['mean'])



joined_data['year'] = pd.to_datetime(joined_data['Date']).dt.year

joined_data['month'] = pd.to_datetime(joined_data['Date']).dt.month

joined_data['weekday'] = pd.to_datetime(joined_data['Date']).dt.weekday_name



#print(joined_data.weekday)



#week_data=joined_data.groupby(['weekday'])['Close'].mean()



mean_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'mean': 'mean'})

std_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'std': np.std})

min_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'min': np.min})

max_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'max': np.max})

median_df = joined_data.groupby(['year','weekday'], as_index=False)['Close'].agg({'median': np.median})







week_data = pd.concat([mean_df, std_df['std'], min_df['min'],max_df['max'], median_df['median']], axis=1)

week_data['var_coeff'] = std_df['std'] / mean_df['mean']



week_data.head(10)
years = [2013,2014,2015,2016,2017]





fig, ax = plt.subplots(len(years),4,sharex=True, sharey=False ,figsize=(12,12))

fig.suptitle('Day of Week Analysis')



for i, year in enumerate(years):

    holder = week_data[week_data['year']==year]

    #print(holder)

    

    my_xticks = np.array(day_of_week.weekday)

    plt.xticks(range(len(holder['mean'])), my_xticks, rotation=90)

    

    ax[i][0].plot(range(len(holder['mean'])), holder['mean'])

   # ax[i][0].set_ylim(min(holder['mean']), max(holder['mean']))

    



    ax[i][1].errorbar(

    range(len(holder['mean'])),     # X

    holder['mean'],    # Y

    yerr=holder['std'],        # Y-errors

      # format line like for plot()

    linewidth=3,   # width of plot line

    elinewidth=1,# width of error bar line

    ecolor='r',    # color of error bar

    capsize=4,     # cap length for error bar

    capthick=2,  # cap thickness for error bar

    )

    



    ax[i][2].plot(range(len(holder['mean'])), holder['mean'])

    #ax[i][2].set_ylim(abs(max(holder['mean']) - max(holder['std'])), abs(max(holder['mean']) + max(holder['std'])))

    

    ax[i][3].errorbar(

    range(len(holder['mean'])),     # X

    holder['mean'],    # Y

    yerr=holder['std'],        # Y-errors

      # format line like for plot()

    linewidth=3,   # width of plot line

    elinewidth=1,# width of error bar line

    ecolor='r',    # color of error bar

    capsize=4,     # cap length for error bar

    capthick=2,  # cap thickness for error bar

    )

    

    

    ax[i][2].set_ylim(0, 4500) #these values are set by experimenting with 'sharey' attribute in subplots() function

                               #the goal is to compare the mean for all years relative to bitcoin increased price now

    ax[i][3].set_ylim(0, 4500)



    #ax[i].set_xlabel("Year"+" "+str(year),fontsize=10)

    #ax[i].set_ylabel("Bitcoin Price",fontsize=10)

    ax[i][0].set_title(year)

    ax[i][1].set_title(year)

    ax[i][2].set_title(year)   

    ax[i][3].set_title(year)  



for tick in ax[i][0].get_xticklabels():

    tick.set_rotation(90)

for tick in ax[i][1].get_xticklabels():

    tick.set_rotation(90)

for tick in ax[i][2].get_xticklabels():

    tick.set_rotation(90)





#plt.savefig('yearly_dayofweek.png')

plt.show()


years = [2013,2014,2015,2016,2017]



fig,ax = plt.subplots(figsize=(12, 8))  #plt.subplots(figsize=(12,8))

fig.suptitle('Month of year - Price Spread')

#fig.subplots_adjust(hspace=.5,wspace=0.7)





for i, year in enumerate(years):

    holder = year_month_data[year_month_data['year']==year]

    #print(holder)

    ax.plot(range(len(holder['std'])), holder['std'], label=str(year))

    ax.set_xlabel("Year",fontsize=10)

    ax.set_ylabel("Price Spread",fontsize=10)

    #ax[i].title("Month of year Price"+" "+str(year), fontsize=10)

    

    

my_xticks = np.array(holder.month)

plt.xticks(range(len(holder['std'])), my_xticks)#Set label location

plt.legend(loc='best')

#plt.savefig('btc_stability.png')

plt.show()
selected_col = joined_data[['Close','btc_market_cap',

                            'btc_avg_block_size',

                            'btc_n_transactions_per_block',

                            'btc_hash_rate',

                            'btc_difficulty',

                            'btc_cost_per_transaction',

                            'btc_n_transactions']]



selected_col.head()

corrmat = selected_col.corr(method='pearson')



columns = ['Close']

my_corrmat = corrmat.copy()

mask = my_corrmat.columns.isin(columns)

my_corrmat.loc[:, ~mask] = 0

#print(my_corrmat)



fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(my_corrmat, annot=False, fmt="f", cmap="Blues") #vmax=1., square=True)

plt.title("Correlation Between Price and other factors", fontsize=15)

#plt.savefig('variablecorrelation.png', bbox_inches='tight')

plt.show()