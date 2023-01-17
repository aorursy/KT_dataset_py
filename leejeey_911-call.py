# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# importing data visualization libraries

import seaborn as sns

sns.set_style('whitegrid')

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read in the csv file as a dataframe called df

df = pd.read_csv('/kaggle/input/montcoalert/911.csv')
df.head()
# top 5 zipcodes and townships

df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
# 'title' is the category of the call

df['title'].value_counts()
# number of unique title codes

# There are 141 reasons of 911 calls

df['title'].nunique()
# creating a new column called 'reason' to save category values from 'title'

df['reason'] = df['title'].apply(lambda x: x[:x.index(':')])
# most common reason for 911 call based on 'reason' column

df['reason'].value_counts()
# visualizing by 'reason' column with countplot

sns.countplot(df['reason'], palette = 'viridis')
# converting data type of 'timeStamp' from str to DateTime objects

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
# adding 3 new columns (hour, month, day of week) to the dataframe based on the 'timeStamp' column

df['hour'] = df['timeStamp'].apply(lambda time:time.hour)

df['month'] = df['timeStamp'].apply(lambda time:time.month)

df['day of week'] = df['timeStamp'].apply(lambda time:time.dayofweek)
df.head()
# changing day of week numerics into actual string values

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}



df['day of week'] = df['day of week'].map(dmap)
df.head()
# countplot of the 'day of week' column with hue of 'reason' column

sns.countplot(x='day of week', data=df, hue='reason', palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# same with month

sns.countplot(x='month', data=df, hue='reason', palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth = df.groupby('month').count()

byMonth
byMonth['twp'].plot()
sns.lmplot(x='month', y='twp', data=byMonth.reset_index())
# creating a new column called 'date' that contains the date from timeStamp column

df['date'] = df['timeStamp'].apply(lambda t: t.date())
# plotting a graph grouped by 'date' column

df.groupby('date').count()['twp'].plot()

plt.tight_layout()
# 3 different plots with each plot representing a reason for the 911 call

# Traffic

df[df['reason']=='Traffic'].groupby('date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()
# Fire

df[df['reason']=='Fire'].groupby('date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
# EMS

df[df['reason']=='EMS'].groupby('date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
df.head()
plt.figure(figsize=(12,4))

plt.xlabel('hour')

plt.title("All Situations by time")

sns.countplot(df['hour'])
plt.figure(figsize=(12,4))

plt.xlabel('month')

plt.title("All Situations by month")

sns.countplot(df['month'])
order = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']

plt.figure(figsize=(12,4))

plt.xlabel('day of week')

plt.title("All Situations by day of week")

sns.countplot(df['day of week'], order=order)
df['twp'].value_counts()
plt.figure(figsize=(18,6))

plt.xlabel('Location')

plt.title("All Situations by Location")

g = sns.countplot(df['twp'])

g

g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()
# reindexing for heatmap

dayHour = df.groupby(by=['day of week','hour']).count()['reason'].unstack()

dayHour.head()
# heatmap

plt.figure(figsize=(12,6))

sns.heatmap(dayHour, cmap='viridis')
# clustermap

sns.clustermap(dayHour, cmap='viridis')
# same with month as the column

dayMonth = df.groupby(by=['day of week', 'month']).count()['reason'].unstack()

plt.figure(figsize=(12,6))

sns.heatmap(dayMonth, cmap='viridis')
# month clustermap

sns.clustermap(dayMonth, cmap='viridis')