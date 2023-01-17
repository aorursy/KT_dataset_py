import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().nlargest(5)
sum(df['zip'].value_counts().nlargest(5))/sum(df['zip'].value_counts()) * 100
df['twp'].value_counts().nlargest(5)
sum(df['twp'].value_counts().nlargest(5))/sum(df['twp'].value_counts()) * 100
df['title'].unique().size

#len(df['title'].unique()) # Another way to get the result
reason = df['title'].apply(lambda x: x.split(':')[0])

df['Reason'] = reason
df['Reason'].value_counts()
sns.countplot(df['Reason'])

#sns.countplot(x='Reason', data= df) #Another way to plot it
type(df['timeStamp'][0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
hour = df['timeStamp'].apply(lambda x: x.hour)

month = df['timeStamp'].apply(lambda x: x.month)

day =  df['timeStamp'].apply(lambda x: x.weekday())

#day = df['timeStamp'].apply(lambda x: x.dayofweek) #Another way tp get the weekday
# As we will got the weekday in number from .weekday(), we could create a dictionary with key-value pair of its name.

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

week_day = day.map(dmap)
df['Hour'] = hour

df['Month'] = month

df['Day of Week'] = week_day
sns.countplot(x = df['Day of Week'],hue=df['Reason']).legend(loc='center left', bbox_to_anchor=(1, 0.5))

sns.countplot(df['Month'],hue=df['Reason']).legend(loc='center left', bbox_to_anchor=(1, 0.5))
byMonth = df.groupby('Month').count()

byMonth

#As we notice before, there are some NA values inside which caused the count() number difference
byMonth['twp'].plot()
byMonth['index'] = byMonth.index

sns.lmplot(x='index', y='twp',data=byMonth)

#sns.lmplot(x='Month', y='twp',data = byMonth.reset_index()) #Another way to plot it
date = df['timeStamp'].apply(lambda x: x.date())

df['Date']= date

#注意TimeStamp object的特性
byDate = df.groupby('Date').count()

byDate['twp'].plot()

plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date')['twp'].count().plot()

plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date')['twp'].count().plot()

plt.tight_layout()
df[df['Reason']=='EMS'].groupby('Date')['twp'].count().plot()

plt.tight_layout()
day_hour = df.groupby(['Day of Week','Hour']).count()['Reason'].unstack()

day_hour
plt.figure(figsize=(12,6))

sns.heatmap(day_hour, cmap="viridis")
sns.clustermap(day_hour,cmap='viridis')
month_hour = df.groupby(['Day of Week','Month']).count()['Reason'].unstack()

month_hour
plt.figure(figsize=(12,6))

sns.heatmap(month_hour,cmap='viridis')
sns.clustermap(month_hour,cmap='viridis')