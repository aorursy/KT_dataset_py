#Exercise https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/

#Raw data here: https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv



#This project is reviewing Germany's wind and solar energy consumption from 2006 - 2017



# Date — The date (yyyy-mm-dd format)

# Consumption — Electricity consumption in GWh

# Wind — Wind power production in GWh

# Solar — Solar power production in GWh

# Wind+Solar — Sum of wind and solar power production in GWh
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(11,4)})

from time import time

import datetime

data = pd.read_csv("../input/Open Power Systems Data.csv")





print(data.head())

print('\n')

print(data.columns)

print('\n')

print(data.info())

print('\n')

print(data.describe())
#Formatting dates

data = pd.read_csv("../input/Open Power Systems Data.csv",index_col=0,parse_dates=True)

data.head()
data['Year'] = data.index.year

data['Month'] = data.index.month

data['Day of Week'] = data.index.weekday_name

data.head()
#Select data of particular day



print(data.loc['2017-08-10'])
#Couple days of data

print(data.loc['2017-08-10':'2017-08-15'])
data['Consumption'].plot()
cols_plot = ['Consumption','Solar','Wind']



axes = data[cols_plot].plot(marker='o', alpha=1, linestyle='None', figsize=(11, 9), subplots=True)



for x in axes:

    x.set_ylabel('Daily Totals (Gwh)')
#Seasonality of data consumption
ax = data.loc['2017', 'Consumption'].plot()

ax.set_ylabel('Daily Consumption (GWh)')

#Looking only at January and February



ax = data.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-')

ax.set_ylabel('Daily Consumption (GWh)')
import matplotlib.dates as mdates



fig, ax = plt.subplots()

ax.plot(data.loc['2017-01':'2017-02', 'Consumption'], marker='o', linestyle='-')

ax.set_ylabel('Daily Consumption (GWh)')

ax.set_title('Jan-Feb 2017 Electricity Consumption')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):

    sns.boxplot(data=data, x='Month', y=name, ax=ax)

    ax.set_ylabel('GWh')

    ax.set_title(name)

    # Remove the automatic x-axis label from all but the bottom subplot

    if ax != axes[-1]:

        ax.set_xlabel('')
sns.boxplot(data=data, x='Year', y='Consumption')

sns.boxplot(data=data, x='Day of Week', y='Consumption')
