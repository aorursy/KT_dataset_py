# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd;

import matplotlib.pyplot as plt

import matplotlib

from datetime import datetime

from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, WeekdayLocator, DayLocator



years = YearLocator()   # every year

months = MonthLocator()  # every month

days = DayLocator()  # every day

yearsFmt = DateFormatter('Y%')

monthsFmt = DateFormatter('%Y-%m')



# Any results you write to the current directory are saved as output.
calendar_data = pd.read_csv("../input/calendar.csv");
#The change in average price of available apartments

available_apartments = calendar_data[(calendar_data.available == 't')]

available_apartments.loc[:, 'price'] = available_apartments.price.replace( '[\$,)]','', regex=True )

available_apartments.loc[:, 'price'] = available_apartments.price.astype('float64')



available_apartments = available_apartments[['date', 'price']]

avg_prices = available_apartments.groupby(['date']).mean()
#Plotting average prices of apartments

dates = pd.to_datetime(avg_prices.index.values, errors='ignore')

dates = dates.astype(datetime)

dates = matplotlib.dates.date2num(dates)



fig, axs = plt.subplots(1, 1)

axs.plot_date(dates, avg_prices.price.tolist(), '-')



# format the ticks

axs.xaxis.set_major_locator(months)

axs.xaxis.set_major_formatter(monthsFmt)

axs.xaxis.set_minor_locator(days)

axs.autoscale_view()

axs.fmt_xdata = DateFormatter('%Y-%m-%d')

axs.grid(True)

axs.set_ylabel('Average price')



fig.adjustable = True

fig.set_size_inches(8, 5) 

fig.autofmt_xdate()

plt.show()
#The busiest days in Seattle, which equals to number of not_available_apartments over all the apartments

#in Seattle

available_apartments = calendar_data[['date', 'available']]

#convert available column to 1 if available and 0 if not to do farther grouping operation

available_clm = available_apartments.apply(lambda row : 1 if row['available'] == 't' else 0, 1)

available_apartments.loc[:,'available'] = available_clm

grouped_data   = available_apartments.groupby([available_apartments.date]).agg(['sum', 'count'])

grouped_data.loc[:, 'avg'] = grouped_data['available']['sum']/grouped_data['available']['count']



#plotting the availability over days

dates = pd.to_datetime(grouped_data.index.values, errors='ignore')

dates = dates.astype(datetime)

dates = matplotlib.dates.date2num(dates)



fig, axs = plt.subplots(1,1)

axs.plot_date(dates, grouped_data.avg.tolist(), '-')



axs.xaxis.set_major_locator(months)

axs.xaxis.set_major_formatter(monthsFmt)

axs.xaxis.set_minor_locator(days)

axs.autoscale_view()

axs.fmt_xdata = DateFormatter('%Y-%m-%d')

axs.grid(True)

axs.set_xlabel('time')

axs.set_ylabel('% availability')



fig.adjustable = True

fig.set_size_inches(8, 5) 

fig.autofmt_xdate()

plt.show()