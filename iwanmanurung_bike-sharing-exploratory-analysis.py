# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import networkx as nx

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q1).csv')

df2 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q2).csv')

df3 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q3).csv')

df4 = pd.read_csv('/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q4).csv')
df1['start'] = pd.to_datetime(df1['trip_start_time'], dayfirst=True)
df2['start'] = pd.to_datetime(df2['trip_start_time'], dayfirst=True)
df3['start'] = pd.to_datetime(df3['trip_start_time'], dayfirst=True)
df4['start'] = pd.to_datetime(df4['trip_start_time'], dayfirst=True)
df = df1.copy()

df = df.append([df2, df3, df4], sort=False)

df = df.copy()
df['hour'] = df['start'].dt.hour.values

df['week'] = df['start'].dt.week.values

df['day'] = df['start'].dt.day.values

df['weekday'] = df['start'].dt.day_name()

df.index = df.start.dt.date.values

df.index.name = 'index'

df['counter'] = 1
df2017 = df.copy()
daily = df2017.day.value_counts().sort_index()



plt.figure(figsize=(15,6))

sns.barplot(x=daily.index, y=daily.values)

plt.box(on=None)

plt.xticks(daily.index-1);

plt.xlabel('Day Index');
'The busiest Date of the year are {}'.format(list(daily.sort_values(ascending=False)[:5].index.values))
week = df2017.week.value_counts().sort_index()



plt.figure(figsize=(15,6))

sns.barplot(x=week.index, y=week.values)

plt.box(on=None)

plt.xticks(week.index-1);

plt.xlabel('Week Index');
'The busiest Week of the year are {}'.format(list(week.sort_values(ascending=False)[:5].index.values))
weekday = df2017.weekday.value_counts(ascending=True)



plt.figure(figsize=(10,6))

order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

sns.barplot(x=weekday.index, y=weekday.values, order=order)

plt.box(on=None)

plt.xlabel('Day of the Week Index');
print('The busiest day of the year is ', list(weekday.sort_values()[:3].index))
period = df2017.groupby('hour').sum()



plt.figure(figsize=(8,6))

sns.barplot(x=period.index, y=period.trip_duration_seconds.values/3600, palette='gnuplot2_r')

plt.box(on=None)

plt.xlabel('Time range');

plt.ylabel('Total trip in Hours');
weekday = df2017.groupby('weekday').sum().sort_values('trip_duration_seconds')



plt.figure(figsize=(8,6))

sns.barplot(x=weekday.index, y=weekday.trip_duration_seconds.values/3600, palette='gnuplot2_r')

plt.box(on=None)

plt.ylabel('Total trip in Hours');
df2017['counter'] = 1

user = df2017.groupby(['user_type']).sum()



plt.figure(figsize=(10,10));

plt.subplot(1,2,1)

plt.pie(user.counter.values, startangle=180, labels=user.index, autopct='%.1f%%', explode=(.05,0));

plt.title('Casual vs Member Trip Number');

plt.xlabel

plt.subplot(1,2,2)

plt.pie(user.trip_duration_seconds.values, startangle=180, labels=user.index, autopct='%.1f%%', explode=(.05,0));

plt.title('Casual vs Member Trip Duration');