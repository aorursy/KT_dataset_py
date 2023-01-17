import math

import calendar



import pandas as pd

import missingno as msno

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm



from pylab import rcParams
google = pd.read_csv('../input/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv')

google
google = pd.read_csv('../input/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv', 

                     index_col = 'Date',

                    )

google.head()
google.index.dtype
google = pd.read_csv('../input/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv', 

                     index_col = 'Date',

                     parse_dates=['Date'], # parse 'Date' column as a datetime type

                    )

google.head()
google.index.dtype
humidity = pd.read_csv('../input/historical-hourly-weather-data/humidity.csv')

humidity
humidity = pd.read_csv('../input/historical-hourly-weather-data/humidity.csv', 

                       index_col = 'datetime')

humidity.head()
humidity.index.dtype
humidity = pd.read_csv('../input/historical-hourly-weather-data/humidity.csv', 

                       index_col = 'datetime',

                       parse_dates = ['datetime'],

                      )

humidity.head()
humidity.index.dtype
google.isna().sum()
humidity.isna().sum()
total = humidity.index.size

null_sum = humidity.isna().sum()[humidity.isna().sum() != 0]



pd.DataFrame(data={'Amount of null values': null_sum,

                   'Percentage of null values': (null_sum / total*100).round(2).astype(str) + '%'},

            ).sort_values(by=['Percentage of null values'])
msno.bar(humidity)
msno.heatmap(humidity, cmap='coolwarm')
msno.matrix(humidity)
plt.figure(figsize=(20,10))

# 13:70 chosen to see humidity for 2 full days

y = np.array(humidity[['Houston', 'Los Angeles']])[13:70] 

x = np.array(humidity.index)[13:70]

plt.plot(x, y)
plt.figure(figsize=(20,10))

# 0:700 chosen to see humidity for 1 full month

y = np.array(humidity[['Houston', 'Los Angeles']])[0:700]

x = np.array(humidity.index)[0:700]

plt.plot(x, y)
plt.figure(figsize=(20,10))

# 0:2100 chosen to see humidity for 3 full months

y = np.array(humidity[['Houston', 'San Francisco']])[0:2100]

x = np.array(humidity.index)[0:2100]

plt.plot(x, y)
humidity.iloc[0] # humidity for first date
humidity.iloc[-1] # humidity for last date
humidity = humidity.fillna(method='bfill')

humidity.head()
humidity.iloc[-1]
humidity = humidity.fillna(method='ffill')

humidity.head()
humidity.isna().sum()
google.duplicated().sum()
humidity.duplicated().sum()
plt.figure(figsize=(20,10))

plt.plot(google.index, google['High'], 'r-.')

plt.title("Google")

plt.xlabel("Time")

plt.ylabel("High stock prices")
google_144_months = google[['High']].groupby(pd.Grouper(freq='M')).sum()

google_144_months = google_144_months.rename(columns={'High': "Total monthly prices"})

google_144_months
google_january = google_144_months.groupby(google_144_months.index.month).groups[1]

google_january
months = [i for i in range(1,13)]

month_names = []

for i in months:

    month_names.append(calendar.month_name[i])

month_names
dicts = {}

keys = month_names

values = ["Hi", "I", "am", "John"]

for i in keys:

        dicts[i] = values[i]

print(dicts)
google_january[1].date().strftime('%Y-%m-%d')
jan_google_dates = []

for i in range(12):

    jan_google_dates.append((google_january[i].date().strftime('%Y-%m-%d')))

jan_google_dates
jan_google_prices = []

for i in jan_google_dates:

    jan_google_prices.append(google_144_months['Total monthly prices'][google_144_months.index==i])

jan_google_prices[0].values[0]
prices = []

for i in range(12):

    prices.append(jan_google_prices[i].values[0])

prices
def index_slice(part):

    full_size = google.index.size

    years = 12

    days_in_year = full_size // years

    if part == 1:

        start = 0

        end = days_in_year

    if part != 1:

        start = index_slice(part-1)[1]

        end = index_slice(part-1)[1] + days_in_year

    return [start, end]
# example of above function

index_slice(1) 
def years_slice(part):

    years = range(2006, 2019) # Google prices starts from 2006 to 2018

    return [years[part-1], years[part]] # part-1 because part constists of 1...12
# example of above function

years_slice(1)
def axes_slice(i):

    if i%2 == 0:

        y = 1

        x = i//2-1

    else:

        y = 0

        x = math.floor(i//2)

    return [x, y]
axes_slice(6)
fig, axs = plt.subplots(6,2, figsize=(15, 10))

for i in range(1,13):

    axs[int(axes_slice(i)[0]),

        int(axes_slice(i)[1])].plot(google.index[

                                        int(index_slice(i)[0]):

                                        int(index_slice(i)[1])], 

                                    google['High'][

                                        int(index_slice(i)[0]):

                                        int(index_slice(i)[1])],

                                   )

    axs[int(axes_slice(i)[0]),

        int(axes_slice(i)[1])].set_title(f'Google in {years_slice(i)[0]}')







for ax in axs.flat:

    ax.set(xlabel='Time', ylabel='High stock prices')



# hide x labels and tick labels for top plots and y ticks for right plots

for ax in axs.flat:

    ax.label_outer()
def three_point_MA(data):

    list_ = [np.nan] # np.nan because we aren't able to calculate MA for first price

    for i in range(1, data.index.size-1):

        ma = (data.iloc[i-1] + data.iloc[i] + data.iloc[i+1])/3

        list_.append(ma)

    list_.append(np.nan) # np.nan because we aren't able to calculate MA for last price

    return list_
google_high_trend = pd.DataFrame(data = {

                        'High': google['High'],

                        'Trend': three_point_MA(google['High']),

})



google_high_trend
google_high_and_var = pd.DataFrame(data = {

    'High': google['High'],

    'Variance': google['High'].var(),

})

google_high_and_var
fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(google_high_trend['High'], c='b', ls='--', label='High')

ax.plot(google_high_trend['Trend'], c='r', label = 'Trend')

plt.legend(loc=2)

plt.show()
def seas_var_around_trend(data):

    list_ = [np.nan] # np.nan because we aren't able to find diff between first item and previous one

    for i in range(1, data.index.size):

        var_around_trend = (data.iloc[i] - google_high_trend['Trend'].iloc[i])

        list_.append(var_around_trend)

    return list_
google_statistics = pd.DataFrame(data = {

                        'High': google['High'],

                        'Trend': three_point_MA(google['High']),

                        'Seasonal variation around trend' : seas_var_around_trend(google['High']),

})



google_statistics
fig, axs = plt.subplots(6,2, figsize=(15, 10))

for i in range(1,13):

    axs[int(axes_slice(i)[0]),

        int(axes_slice(i)[1])].plot(google.index[

                                        int(index_slice(i)[0]):

                                        int(index_slice(i)[1])], 

                                    google_statistics['Seasonal variation around trend'][

                                        int(index_slice(i)[0]):

                                        int(index_slice(i)[1])],

                                   )

    axs[int(axes_slice(i)[0]),

        int(axes_slice(i)[1])].set_title(f'Seasonal variation around trend of Google\'s prices in {years_slice(i)[0]}')







for ax in axs.flat:

    ax.set(xlabel='Time', ylabel='Seasonal var around trend')



# hide x labels and tick labels for top plots and y ticks for right plots

for ax in axs.flat:

    ax.label_outer()
def daily_perc_change(data):

    list_ = [np.nan] # np.nan because we aren't able to find diff between first item and previous one

    for i in range(1, data.index.size):

        daily_perc_change = (data.iloc[i]-data.iloc[i-1])/data.iloc[i-1]

        list_.append(daily_perc_change)

    return list_
google_statistics = pd.DataFrame(data = {

                        'High': google['High'],

                        'Daily % change': daily_perc_change(google['High']),

})



google_statistics
fig, axs = plt.subplots(6,2, figsize=(15, 10))

for i in range(1,13):

    axs[int(axes_slice(i)[0]),

        int(axes_slice(i)[1])].plot(google.index[

                                        int(index_slice(i)[0]):

                                        int(index_slice(i)[1])], 

                                    google_statistics['Daily % change'][

                                        int(index_slice(i)[0]):

                                        int(index_slice(i)[1])],

                                   )

    axs[int(axes_slice(i)[0]),

        int(axes_slice(i)[1])].set_title(f'Daily percent change of Google\'s prices in {years_slice(i)[0]}')







for ax in axs.flat:

    ax.set(xlabel='Time', ylabel='% Change')



# hide x labels and tick labels for top plots and y ticks for right plots

for ax in axs.flat:

    ax.label_outer()
rcParams['figure.figsize'] = 11, 9

decomposed_google_volume = sm.tsa.seasonal_decompose(google["High"], 

                                                     period=360,

                                                     model='additive',

                                                    ) 

figure = decomposed_google_volume.plot()

plt.show()
rcParams['figure.figsize'] = 11, 9

decomposed_google_volume = sm.tsa.seasonal_decompose(google["High"], 

                                                     period=360,

                                                     model='multiplicative',

                                                    ) 

figure = decomposed_google_volume.plot()

plt.show()