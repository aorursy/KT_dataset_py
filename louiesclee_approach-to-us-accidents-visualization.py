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

import calendar

sns.set()
month_names = dict(enumerate(calendar.month_abbr))

day_names = dict(enumerate(calendar.day_name))
df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')

df.head()
df.columns
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

df['End_Time'] = pd.to_datetime(df['End_Time'])

df['Year'] = df['Start_Time'].dt.year

df['Hour'] = df['Start_Time'].dt.hour

df['Day'] = df['Start_Time'].dt.dayofweek

df['DayName'] = df['Start_Time'].dt.weekday_name

df['Month'] = df['Start_Time'].dt.month
df = df[(df['Year']>2015) & (df['Year']<2020)]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

fig.tight_layout(pad=3)



hours = df.groupby('Hour').size()

ax1.bar(hours.index, hours)

ax1.set_xticks(hours.index)

ax1.set_xlabel('Factor (Hour)')



days = df.groupby('Day').size()

ax2.bar(days.index, days)

ax2.set_xticks(days.index)

ax2.set_xticklabels([day_names[i] for i in days.index])

ax2.set_xlabel('Factor (Day)')



months = df.groupby('Month').size()

ax3.bar(months.index, months)

ax3.set_xticks(months.index)

ax3.set_xticklabels([month_names[i] for i in months.index])

ax3.set_xlabel('Factor (Month)')



years = df.groupby('Year').size()

ax4.bar(years.index, years)

ax4.set_xticks(years.index)

ax4.set_xlabel('Factor (Year)')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

fig.tight_layout(pad=3)



severity_1 = df[df['Severity']==1]

sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=severity_1, hue='Severity',linewidth=0, ax=ax1)



severity_2 = df[df['Severity']==2]

sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=severity_2, linewidth=0,ax=ax2)



severity_3 = df[df['Severity']==3]

sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=severity_3,linewidth=0,ax=ax3)



severity_4 = df[df['Severity']==4]

sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=severity_4,linewidth=0,ax=ax4)
plt.figure(figsize=(18, 8))

sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=df, linewidth=0, hue='State')
plt.figure(figsize=(14, 8))

sns.countplot(x='State', data=df, order=df['State'].value_counts().iloc[:10].index)
plt.figure(figsize=(14, 8))

sns.countplot(y='Weather_Condition', data=df, order=df['Weather_Condition'].value_counts().iloc[:10].index)