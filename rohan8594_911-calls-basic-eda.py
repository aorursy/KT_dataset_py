import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().head()
df['twp'].value_counts().head()
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

df.head()
df['Reason'].value_counts()
sns.countplot(x='Reason',data=df,palette='viridis')
type(df.loc[0,'timeStamp'])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['Month'] = df['timeStamp'].apply(lambda x: x.month)

df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)
df['Day of Week'].head()
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='Month',data=df,palette='viridis',hue='Reason')

plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
byMonth = df.groupby('Month').count()

byMonth.head()
byMonth.plot.line(x=byMonth.index,y='e')
byMonth.reset_index(inplace=True)

sns.lmplot(x='Month',y='e',data=byMonth)
df['Date'] = df['timeStamp'].apply(lambda x: x.date())
bydate = df.groupby('Date').count()

bydate.plot.line(x=bydate.index,y='e',figsize=(10,4))

plt.tight_layout()
bydate_traffic = df[df['Reason'] == 'Traffic'].groupby('Date').count()

bydate_traffic.plot.line(x=bydate_traffic.index,y='e',figsize=(12,4))

plt.title('Traffic')
bydate_fire = df[df['Reason'] == 'Fire'].groupby('Date').count()

bydate_fire.plot.line(x=bydate_fire.index,y='e',figsize=(12,4))

plt.title('Fire')
bydate_ems = df[df['Reason'] == 'EMS'].groupby('Date').count()

bydate_ems.plot.line(x=bydate_ems.index,y='e',figsize=(12,4))

plt.title('EMS')
bydayhour = df.groupby(by=['Day of Week','Hour']).count()['e'].unstack()

bydayhour.head()
plt.figure(figsize=(12,7))

sns.heatmap(bydayhour,cmap='viridis')
sns.clustermap(bydayhour,cmap='viridis')
bydaymonth = df.groupby(by=['Day of Week','Month']).count()['e'].unstack()

bydaymonth.head()
plt.figure(figsize=(12,7))

sns.heatmap(bydaymonth,cmap='viridis')
sns.clustermap(bydaymonth,cmap='viridis')