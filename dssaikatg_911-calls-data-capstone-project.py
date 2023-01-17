# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the dataset

df = pd.read_csv("../input/911.csv")
df.info()
df.head()
# Top zip-codes for 911 calls

df['zip'].value_counts().iloc[:5]
# Top townships for 911 calls

df['twp'].value_counts().iloc[:5]
# Total no. of titles

df['title'].nunique()
# Create another column for the 'Categorical Reason' for 911 call

df['reason'] = df['title'].apply(lambda x: x.split(': ')[0])

print(df['reason'].unique())

df.head()
# Counts of 'Reason' as available in the dataset

df['reason'].value_counts()
sns.countplot(x='reason', data=df)
type(df['timeStamp'].iloc[0])
# Change the datatype of the 'Time Stamp' column to 'datetime'

df['timeStamp'] = df['timeStamp'].apply(lambda x: pd.to_datetime(x))

type(df['timeStamp'].iloc[0])
# Extract specifics from the 'timeStamp' column for further exploration

df['hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['month'] = df['timeStamp'].apply(lambda x: x.month)

df['day_of_week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
print(df['day_of_week'].unique())

df.head()
# Countplot showing the day-of-week with separate bars denoting the 'reason'

sns.countplot(x='day_of_week', data=df, hue='reason')
# Countplot showing the month with separate bars denoting the 'reason'

sns.countplot(x='month', data=df, hue='reason')
# Plot denoting no. of calls by month

bymonth = df.groupby('month').count()

print(bymonth.head())

sns.pointplot(x='month', y='e', data=bymonth.reset_index())
sns.lmplot(x='month', y='e', data=bymonth.reset_index())
# Extract and create a separate column for the date info

df['date'] = df['timeStamp'].apply(lambda x: x.date())

df.head()
# Plot showing the counts of calls based on the dates

plt.figure(figsize=(10, 5))

df.groupby('date').count()['e'].plot()
# Plot showing the counts of calls for Traffic use cases based on the dates

plt.figure(figsize=(10, 5))

plt.title('Traffic')

df[df['reason'] == 'Traffic'].groupby('date').count()['e'].plot()
# Plot showing the counts of calls for Fire use cases based on the dates

plt.figure(figsize=(10, 5))

plt.title('Fire')

df[df['reason'] == 'Fire'].groupby('date').count()['e'].plot()
# Plot showing the counts of calls for EMS use cases based on the dates

plt.figure(figsize=(10, 5))

plt.title('EMS')

df[df['reason'] == 'EMS'].groupby('date').count()['e'].plot()
# Create another column for the day name, extracted out of the timeStamp column

df['day_name'] = df['timeStamp'].apply(lambda x: x.day_name())

df.head()
# Heatmap displaying the association between day_name and hours

countby_day_hour = df.groupby(by=['day_name', 'hour']).count().unstack(level=-1)['lat']

plt.figure(figsize=(10, 4))

sns.heatmap(countby_day_hour)
# Clustermap displaying the association between day_name and hours

plt.figure(figsize=(10, 6))

sns.clustermap(countby_day_hour)
# Heatmap displaying the association between day_name and month

countby_day_month = df.groupby(by=['day_name', 'month']).count().unstack(level=-1)['lat']

plt.figure(figsize=(10, 4))

sns.heatmap(countby_day_month)
# Clustermap displaying the association between day_name and month

plt.figure(figsize=(10, 6))

sns.clustermap(countby_day_month)