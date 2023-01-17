# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head(10)
print("Length of the dataset: %d\nNumber of different items: %d" % (len(df), len(df.Item.unique())))

print("Number of different transactions: %d" % df['Transaction'].max())
nTransactions = df.groupby(['Transaction'])['Item'].count().reset_index()
nTransactions.columns = ['Transaction', 'nItems']
#Checking...
nTransactions.head()

print("Number of Items per transaction in average: %.2f" % nTransactions['nItems'].mean())
nTransactions['nItems'].describe()
#Transforming original dataset. First, adding the new column "number of Items, nItems" to the original. Just joining by Transaction .
# tt stands for "transformed Transactions" :o)
#On the other hand, we build 'times' dataset for time analysis

tt = nTransactions.merge(df, on='Transaction')
times = tt.drop_duplicates(subset='Transaction', keep='first')[['Transaction', 'nItems', 'Date', 'Time']]
times.head()
timeCount = times.groupby('Time').count()['Date'].reset_index()
timeCount.head()
#Ok, let's group by a more generic hour. What about hour and decimal of minutes? ;)

MIN_UNIT_INDEX=4
HOUR_UNIT_INDEX=2
times['hourMin'] = times['Time'].apply(lambda x: str(x)[:HOUR_UNIT_INDEX]+"x")
times.head()
timeCount = times.groupby('hourMin').count()['Date'].reset_index()
timeCount.columns=['hourMin', 'countTransactions']

timeCount.head(10)
len(timeCount)
timeCount.describe()
timeCount.plot.bar(x='hourMin', y='countTransactions', rot=30, figsize=(15,10))
times['dayWeek'] = pd.to_datetime(times['Date']).dt.weekday_name
times.head()
timesDay = times.groupby('dayWeek').count()['Transaction'].reset_index()
timesDay.columns = ['dayWeek', 'nTransactions']

timesDay.plot.bar(x='dayWeek', y='nTransactions', rot=30, figsize=(15,10))
times = times[['Transaction', 'nItems', 'hourMin', 'dayWeek']]
dfplus = df.merge(times, on='Transaction')
dfplus.head()
res = dfplus.groupby(['dayWeek', 'Item']).count()['nItems'].reset_index()
res
mostPerDay = res.groupby('dayWeek').agg(['min', 'max'])
mostPerDay
totalItems = res.groupby('Item').sum().sort_values(by='nItems', ascending=False)
# Top 10 ;P
totalItems[:10]
