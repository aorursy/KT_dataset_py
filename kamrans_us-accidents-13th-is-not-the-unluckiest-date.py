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

sns.set()
df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
print(df.info())
counts_by_severity = df.groupby('Severity').count()['ID']
counts_by_severity.plot(kind='bar')
counts_by_severity.plot(kind='pie')
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

df['Year'] = df['Start_Time'].dt.year

df['Month'] = df['Start_Time'].dt.month_name()

df['Day'] = df['Start_Time'].dt.day

df['Day_name'] = df['Start_Time'].dt.day_name()

df['Hour'] = df['Start_Time'].dt.hour
accidents_by_hour = df.groupby('Hour').count()['ID']

accidents_by_hour.plot(kind='bar', figsize=(10, 8))
weekdays_df = df[(df['Day_name'] == 'Monday') | (df['Day_name'] == 'Tuesday') | (df['Day_name'] == 'Wednesday')

                 | (df['Day_name'] == 'Thursday') | (df['Day_name'] == 'Friday')]
weekday_acc_by_hour = weekdays_df.groupby('Hour').count()['ID']

weekday_acc_by_hour.plot(kind='bar', figsize=(10, 8))
weekend_df = df[(df['Day_name'] == 'Saturday') | (df['Day_name'] == 'Sunday')]

weekend_acc_by_hour = weekend_df.groupby('Hour').count()['ID']

weekend_acc_by_hour.plot(kind='bar', figsize=(10, 8))
accidents_day_of_week = df.groupby('Day_name').count()['ID'].sort_values(ascending=False)

accidents_day_of_week.plot(kind='bar')
accidents_day_of_week.plot(kind='pie')
acc_by_day_of_month = df.groupby('Day').count()['ID']

acc_by_day_of_month.plot(kind='bar', figsize=(12, 6))
accidents_by_month = df.groupby('Month').count()['ID'].sort_values(ascending=False)

accidents_by_month.plot(kind='bar')
accidents_by_month.plot(kind='pie')
accidents_by_year = df.groupby('Year').count()['ID'].drop([2015, 2020])

accidents_by_year.plot(kind='bar')
plt.figure(figsize=(10, 8))

sns.scatterplot(x='Start_Lng', y='Start_Lat', hue='Severity', data=df)