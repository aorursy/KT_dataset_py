import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/911-dataset/911.csv')
df.info()
df.head()
df['zip'].head()
df['twp'].head()
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
df['Reason'].value_counts()
sns.set()
sns.countplot('Reason', data=df)
type(df['timeStamp'][0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
time = df['timeStamp'].iloc[0]
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
dmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thur',4:'Fri',5:'sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
df.head()
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
monthwise=df.groupby('Month').count()
monthwise.head()
monthwise['twp'].plot()
sns.lmplot(x='Month',y='twp',data=monthwise.reset_index())
df['Date'] = df['timeStamp'].apply(lambda t: t.date())
df.head()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()
plt.figure(figsize=(14,8))
sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()
plt.figure(figsize=(14,8))
sns.heatmap(dayMonth,cmap='viridis')
sns.clustermap(dayMonth,cmap='viridis')