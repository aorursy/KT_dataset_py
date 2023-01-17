import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

from pandas import DataFrame 

from datetime import datetime

from dateutil.relativedelta import relativedelta

import time

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_uber= pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jul14.csv')
df_uber.head()
df_uber.isnull().sum()
df_uber['Date/Time']=df_uber['Date/Time'].map(pd.to_datetime)
df_uber.info()
def get_DateOfMonth(dt):

    return dt.day 

def get_weekday(dt):

    return dt.dayofweek 

def get_hour(dt):

    return dt.hour

def get_weekday_name(dt):

    return dt.day_name()

df_uber['DOM'] =df_uber['Date/Time'].map(get_DateOfMonth)

df_uber['Weekday'] =df_uber['Date/Time'].map(get_weekday)

df_uber['Hour'] =df_uber['Date/Time'].map(get_hour)

df_uber['DayOfWeek'] =df_uber['Date/Time'].map(get_weekday_name)
df_uber.head()
df_pivot_hour = df_uber.pivot_table(index=['Weekday','DayOfWeek'],

                                  values='Base',

                                  aggfunc='count')

df_pivot_hour.plot(kind='bar', figsize=(8,6))

plt.ylabel('Day of the week Frequency')

plt.title('Journeys by Week Day');
# An "interface" to matplotlib.axes.Axes.hist() method

n, bins, patches = plt.hist(df_uber.DOM.sort_values(), bins='auto', color='#0504aa',

                            alpha=0.7, rwidth=3.1 , range=(0.5,30.5))

plt.grid(axis='y', alpha=0.75)

plt.xlabel('Date of Month')

plt.ylabel('Frequency')

plt.title('Frequency of DOM by Uber-July14')

plt.text(23, 45, r'$\mu=15, b=3$')
n, bins, patches = plt.hist(df_uber.Hour, bins='auto', color='#0504aa',

                            alpha=0.7, rwidth=3.1 , range=(0.5,24))

plt.grid(axis='y', alpha=0.75)

plt.xlabel('Date of Month')

plt.ylabel('Frequency')

plt.title('Frequency of DOM by Uber-July14')

plt.text(23, 45, r'$\mu=15, b=3$')
df_uber['Lat'].hist(bins=100, range=(40.4,41.1))
df_uber['Lon'].hist(bins=100, range=(-74.2,-73.7))
f, ax = plt.subplots(figsize=(8,6))

ax = sns.kdeplot(pd.Series(df_uber['DOM'], name="Day Of Month"),shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

ax = sns.kdeplot(pd.Series(df_uber['Hour'], name="Hour"),shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

ax = sns.kdeplot(pd.Series(df_uber['Lat'], name="Lat"),shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

ax = sns.kdeplot(pd.Series(df_uber['Lon'], name="Lon"),shade=True, color='r')

plt.show()