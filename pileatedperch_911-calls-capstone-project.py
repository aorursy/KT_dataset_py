import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/911.csv')
df.drop(labels = 'e',axis=1,inplace=True)
df.info()
df.head()
df['zip'].value_counts().iloc[:5]
df['twp'].value_counts().iloc[:5]
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda s:s.split(':')[0])
df['Reason'].head()
df['Reason'].value_counts()
sns.countplot(x='Reason', data=df)
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
df['Hour'] = df['timeStamp'].apply(lambda time:time.hour)
df['Month'] = df['timeStamp'].apply(lambda time:time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time:time.dayofweek)
df.sample()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].apply(lambda int:dmap[int])
sns.countplot(x='Day of Week', hue='Reason', data=df)
plt.legend(bbox_to_anchor=(1,1))
sns.countplot(x='Month', hue='Reason', data=df)
plt.legend(bbox_to_anchor=(1,1))
byMonth = df.groupby(by='Month').count()
byMonth
byMonth['lat'].plot()
sns.pointplot(x=byMonth.index, y = 'lat', data=byMonth, markers='.')
byMonth['Month'] = byMonth.index
byMonth
sns.lmplot(x='Month', y='lat', data=byMonth)
df['Date'] = df['timeStamp'].apply(lambda time:time.date())
df.head()
df.groupby(by='Date').count()['lat'].plot()
plt.tight_layout()
df[df['Reason']=='Traffic'].groupby(by='Date').count()['lat'].plot()
plt.title('Traffic')
df[df['Reason']=='Fire'].groupby(by='Date').count()['lat'].plot()
plt.title('Fire')
df[df['Reason']=='EMS'].groupby(by='Date').count()['lat'].plot()
plt.title('EMS')
dfGrid = df.groupby(by=['Day of Week','Hour']).count()['lat'].unstack()
dfGrid = dfGrid.loc[['Sun','Mon','Tue','Wed','Thu','Fri','Sat']]
dfGrid
plt.figure(figsize=(12,6))
sns.heatmap(dfGrid, cmap='viridis')
sns.clustermap(dfGrid, cmap='viridis')
dfMonth = df.groupby(['Day of Week','Month']).count()['lat'].unstack()
dfMonth = dfMonth.loc[['Sun','Mon','Tue','Wed','Thu','Fri','Sat']]
dfMonth
plt.figure(dpi=100)
sns.heatmap(dfMonth, cmap='viridis')
plt.figure(dpi=100)
sns.clustermap(dfMonth, cmap='viridis')