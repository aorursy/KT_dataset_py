import os



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, ArtistAnimation

import seaborn as sns
stocks_df = pd.read_csv('../input/reliance-stock-datajul-2019-jul-2020/Stock-Data-Reliance.csv')

stocks_df.head()
stocks_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

stocks_df.head()
null = pd.DataFrame(stocks_df.isnull().sum(axis = 0), columns = ['total null vals'])

null['percent_null'] = null['total null vals']/len(stocks_df)

null
stocks_df[stocks_df.High.isnull()]
stocks_df.dropna(axis = 0, inplace = True)

stocks_df.head()
stocks_df.dtypes.to_frame()
# converting Date column to date-time format

stocks_df.Date = pd.to_datetime(stocks_df.Date)





# converting Open, High, Low, Close, Adj Close, Volume to floating points

def floatify(x):

    try:

        x = float(x.replace(',',''))

    except:

        x = np.nan

    return x



cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']



for col in cols:

    stocks_df[col] = stocks_df[col].apply(lambda x: floatify(x))



# dropping the newly generated NaN values

stocks_df.dropna(axis = 0, inplace = True)

stocks_df.head()
# reversing the dataframe

stocks_df = stocks_df.iloc[::-1]



# resetting the index

stocks_df.reset_index(drop=True, inplace=True)



stocks_df.head()
sns.set_style('darkgrid')

fig,ax = plt.subplots(figsize = (16,8)) 

sns.lineplot(x = 'Date', y = 'Adj Close', data = stocks_df, ax = ax)

fig.show()
# extracting features from the given data

stocks_df['month'] = pd.DatetimeIndex(stocks_df['Date']).month

stocks_df['year'] = pd.DatetimeIndex(stocks_df['Date']).year

stocks_df['day'] = pd.DatetimeIndex(stocks_df['Date']).day

stocks_df['weekday'] = pd.DatetimeIndex(stocks_df['Date']).weekday

stocks_df['Avg'] = (stocks_df.High + stocks_df.Low)/2
monthly_data = stocks_df.groupby(['year', 'month'])

month_list = []

mean_price = []

for k,_ in monthly_data:

    month = str(k[1]) + '-' + str(k[0])

    month_list.append(month)

    mean_price.append(monthly_data.get_group(k).mean()['Adj Close'])



fig,ax = plt.subplots(figsize = (10, 8)) 

sns.barplot(y = month_list, x = mean_price, ax = ax)

ax.set_xlabel('Mean Adjusted Closing Price')

ax.set_ylabel('Month')

ax.set_title('AVERAGE STOCK PRICE PER MONTH')

fig.show()
def dailyPriceFluctuations(grouped_df, month, year):

    data = grouped_df.get_group((year,month))

    month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    fig, ax = plt.subplots(figsize = (15,5))

    sns.lineplot(x = 'day', y = 'High', marker = 'o', data = data, ax = ax)

    sns.lineplot(x = 'day', y = 'Low', marker = 's', data = data, ax = ax)

    ax.set_title('MONTH: ' + month_dict[month] + '-' + str(year))

    ax.set_ylabel('Prices')

    ax.set_xlabel('Day of Month')

    fig.show()
# July, 2019

dailyPriceFluctuations(monthly_data, month = 7, year = 2019)
# December, 2019

dailyPriceFluctuations(monthly_data, month = 12, year = 2019)
# March, 2020

dailyPriceFluctuations(monthly_data, month = 3, year = 2020)
# July, 2020

dailyPriceFluctuations(monthly_data, month = 7, year = 2020)
days = np.array(['Mon', 'Tues', 'Wed', 'Thru', 'Fri'])



fig, ax = plt.subplots(1, 4, figsize = (20, 4))



week_data = stocks_df.groupby(['weekday']).mean()

avg_daily = ((week_data.High + week_data.Low)/2).to_numpy()

sns.barplot(x = days, y = avg_daily, ax = ax[0])

ax[0].set_ylim(1400, 1450)

ax[0].set_title('Average Daily Data')



apr_data = monthly_data.get_group((2020,4)).groupby(['weekday']).mean()

avg_daily = ((apr_data.High + apr_data.Low)/2).to_numpy()

sns.barplot(x = days, y = avg_daily, ax = ax[1])

ax[1].set_ylim(1200, 1350)

ax[1].set_title('April-2020 Daily Data')





may_data = monthly_data.get_group((2020,5)).groupby(['weekday']).mean()

avg_daily = ((may_data.High + may_data.Low)/2).to_numpy()

sns.barplot(x = days, y = avg_daily, ax = ax[2])

ax[2].set_ylim(1400, 1500)

ax[2].set_title('May-2020 Daily Data')





jun_data = monthly_data.get_group((2020,6)).groupby(['weekday']).mean()

avg_daily = ((jun_data.High + jun_data.Low)/2).to_numpy()

sns.barplot(x = days, y = avg_daily, ax = ax[3])

ax[3].set_ylim(1600, 1680)

ax[3].set_title('June-2020 Daily Data')





fig.show()
may_data = monthly_data.get_group((2020,5))

volume = may_data.Volume/100000



fig, ax = plt.subplots(1, 4, figsize = (20, 4))





volume = stocks_df.Volume/100000

sns.regplot(x = stocks_df['Avg'], y = volume, ax = ax[0])

ax[0].set_ylabel('Volume(1= 1e5)')

ax[0].set_title('Overall Data')



dec_data = monthly_data.get_group((2019,12))

volume = dec_data.Volume/100000

sns.regplot(x = dec_data['Avg'], y = volume, ax = ax[1])

ax[1].set_title('Dec-2019')

ax[1].set_ylabel('')



may_data = monthly_data.get_group((2020,5))

volume = may_data.Volume/100000

sns.regplot(x = may_data['Avg'], y = volume, ax = ax[2])

ax[2].set_title('May-2020')

ax[2].set_ylabel('')



jun_data = monthly_data.get_group((2020,6))

volume = jun_data.Volume/100000

sns.regplot(x = jun_data['Avg'], y = volume, ax = ax[3])

ax[3].set_title('Jun-2020')

ax[3].set_ylabel('')





fig.show()

# may_data