# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option('display.max_columns',55)
pd.set_option('display.max_rows',1500)
data = pd.read_csv('/kaggle/input/pump-sensor-data/sensor.csv')
data.info()
del data['Unnamed: 0']
data.head(15)
data.index = data['timestamp']
data.index = pd.to_datetime(data.index)
del data['timestamp']
data.describe()
corr  = data.corr()
fig, ax = plt.subplots(figsize=(8,8))  

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

    

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

)
data['machine_status'].unique()
data[(data['machine_status'] == 'BROKEN')]
data[(data['machine_status'] == 'RECOVERING')]
data[(data['machine_status'] == 'RECOVERING')].info()
columns = ['sensor_00','sensor_06','sensor_07','sensor_08','sensor_09','sensor_51']
for column in columns:

    print('{0} Original'.format(column))

    display(data[(data['machine_status'] == 'NORMAL')][column].describe())

    print('{0} In Recovery'.format(column))

    display(data[(data['machine_status'] == 'RECOVERING')][column].describe())
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation=70)

ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_00'],marker='o', linestyle='-')

plt.grid(True) 

ax.set_ylabel('Reading Unit')

ax.set_title('First Broken: Sensor_00 Reading')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.MinuteLocator())

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));

fig, ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation=70)

ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_06'],marker='o', linestyle='-')

plt.grid(True) 

ax.set_ylabel('Reading Unit')

ax.set_title('First Broken: Sensor_06 Reading')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.MinuteLocator())

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));

fig, ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation=70)

ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_07'],marker='o', linestyle='-')

plt.grid(True) 

ax.set_ylabel('Reading Unit')

ax.set_title('First Broken: Sensor_07 Reading')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.MinuteLocator())

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));

fig, ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation=70)

ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_08'],marker='o', linestyle='-')

plt.grid(True) 

ax.set_ylabel('Reading Unit')

ax.set_title('First Broken: Sensor_08 Reading')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.MinuteLocator())

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));
fig, ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation=70)

ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_09'],marker='o', linestyle='-')

plt.grid(True) 

ax.set_ylabel('Reading Unit')

ax.set_title('First Broken: Sensor_09 Reading')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.MinuteLocator())

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));

fig, ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation=70)

ax.plot(data.loc['2018-04-12 21:00:00':'2018-04-12 22:15:00', 'sensor_51'],marker='o', linestyle='-')

plt.grid(True) 

ax.set_ylabel('Reading Unit')

ax.set_title('First Broken: Sensor_51 Reading')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.MinuteLocator())

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'));

#Getting the list of columns

cols = list(data.columns)
#Removing sensor 15 as it is completely null

cols.remove('sensor_15')
for i in cols:

    fig, ax = plt.subplots(figsize=(18,5))

    plt.xticks(rotation=90)

    ax.plot(data.loc['2018-04-12 12:00:00':'2018-04-14 12:00:00', i],marker='o', linestyle='-')

    plt.grid(True) 

    ax.set_ylabel('Reading Unit')

    ax.set_title('First Broken: {0} Reading'.format(i))

    # Set x-axis major ticks to weekly interval, on Mondays

    ax.xaxis.set_major_locator(mdates.HourLocator())

    # Format x-tick labels as 3-letter month name and day number

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))
for i in cols:

    fig, ax = plt.subplots(figsize=(18,5))

    plt.xticks(rotation=90)

    ax.plot(data.loc['2018-04-17 12:00:00':'2018-04-19 12:00:00', i],marker='o', linestyle='-')

    plt.grid(True) 

    ax.set_ylabel('Reading Unit')

    ax.set_title('Second Broken: {0} Reading'.format(i))

    # Set x-axis major ticks to weekly interval, on Mondays

    ax.xaxis.set_major_locator(mdates.HourLocator())

    # Format x-tick labels as 3-letter month name and day number

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))
for i in cols:

    fig, ax = plt.subplots(figsize=(18,5))

    plt.xticks(rotation=90)

    ax.plot(data.loc['2018-05-18 20:00:00':'2018-05-20 20:00:00', i],marker='o', linestyle='-')

    plt.grid(True) 

    ax.set_ylabel('Reading Unit')

    ax.set_title('Third Broken: {0} Reading'.format(i))

    # Set x-axis major ticks to weekly interval, on Mondays

    ax.xaxis.set_major_locator(mdates.HourLocator())

    # Format x-tick labels as 3-letter month name and day number

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))
for i in cols:

    fig, ax = plt.subplots(figsize=(18,5))

    plt.xticks(rotation=90)

    ax.plot(data.loc['2018-05-24 12:00:00':'2018-05-26 12:00:00', i],marker='o', linestyle='-')

    plt.grid(True) 

    ax.set_ylabel('Reading Unit')

    ax.set_title('Fourth Broken: {0} Reading'.format(i))

    # Set x-axis major ticks to weekly interval, on Mondays

    ax.xaxis.set_major_locator(mdates.HourLocator())

    # Format x-tick labels as 3-letter month name and day number

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))
for i in cols:

    if i == 'sensor_51':

        continue

    fig, ax = plt.subplots(figsize=(18,5))

    plt.xticks(rotation=90)

    ax.plot(data.loc['2018-06-28 12:00:00':'2018-06-30 12:00:00', i],marker='o', linestyle='-')

    plt.grid(True) 

    ax.set_ylabel('Reading Unit')

    ax.set_title('Fifth Broken: {0} Reading'.format(i))

    # Set x-axis major ticks to weekly interval, on Mondays

    ax.xaxis.set_major_locator(mdates.HourLocator())

    # Format x-tick labels as 3-letter month name and day number

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))
for i in cols:

    fig, ax = plt.subplots(figsize=(18,5))

    plt.xticks(rotation=90)

    ax.plot(data.loc['2018-07-07 12:00:00':'2018-07-09 12:00:00', i],marker='o', linestyle='-')

    plt.grid(True) 

    ax.set_ylabel('Reading Unit')

    ax.set_title('Sixth Broken: {0} Reading'.format(i))

    # Set x-axis major ticks to weekly interval, on Mondays

    ax.xaxis.set_major_locator(mdates.HourLocator())

    # Format x-tick labels as 3-letter month name and day number

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))
for i in cols:

    if i == 'sensor_50':

        continue

    fig, ax = plt.subplots(figsize=(18,5))

    plt.xticks(rotation=90)

    ax.plot(data.loc['2018-07-24 12:00:00':'2018-07-26 12:00:00', i],marker='o', linestyle='-')

    plt.grid(True) 

    ax.set_ylabel('Reading Unit')

    ax.set_title('Seventh Broken: {0} Reading'.format(i))

    # Set x-axis major ticks to weekly interval, on Mondays

    ax.xaxis.set_major_locator(mdates.HourLocator())

    # Format x-tick labels as 3-letter month name and day number

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))
fig, ax = plt.subplots(figsize=(18,5))

plt.xticks(rotation=90)

ax.plot(data.loc['2018-04-17 12:00:00':'2018-04-20 12:00:00', 'machine_status'],marker='o', linestyle='-')

plt.grid(True) 

ax.set_ylabel('Reading Unit')

ax.set_title('Second Broken: machine_status Reading')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.HourLocator())

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))


fig, ax = plt.subplots(figsize=(18,5))

plt.xticks(rotation=90)

ax.plot(data.loc['2018-06-28 12:00:00':'2018-07-06 12:00:00', 'machine_status'],marker='o', linestyle='-')

plt.grid(True) 

ax.set_ylabel('Reading Unit')

ax.set_title('Fifth Broken: machine_status Reading')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.DayLocator())

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))