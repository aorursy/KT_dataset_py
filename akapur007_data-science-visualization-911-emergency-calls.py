import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
df = pd.read_csv('../input/911.csv')
df.head()
df.info()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
uniq = df['title'].unique()
uniq.size
def fun(x):
    print(x.split(':')[0])
df ['reason']=df['title'].apply(lambda x:x.split(':')[0])
df.head()
df['reason'].value_counts()
import seaborn as sns
sns.countplot(x='reason',data=df)
df['timeStamp'].dtype
time = pd.to_datetime(df['timeStamp']).iloc[0]
df['hour']= df['timeStamp'].apply(lambda x: pd.to_datetime(x).hour)

df['month']= df['timeStamp'].apply(lambda x: pd.to_datetime(x).month)

df['day']= df['timeStamp'].apply(lambda x: pd.to_datetime(x).dayofweek)
df.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day']=df['day'].map(dmap)
df.head()
sns.countplot(x='day',data=df, hue='reason')
sns.countplot(x='month',data=df, hue='reason')
groupBy = df.groupby(by='month',as_index='month')
ndf = groupBy.count()
ndf.head()
ndf['twp'].plot()
sns.lmplot(x='month',y='twp',data=ndf.reset_index())
df['date'] = df['timeStamp'].apply(lambda x:pd.to_datetime(x).date())
df.head()
plt.figure(figsize=(15,9))
df.groupby('date').count()['twp'].plot()
plt.tight_layout()
dayHour = df.groupby(by=['day','hour']).count()['reason'].unstack()
dayHour.head()
plt.figure()
sns.heatmap(dayHour)
plt.figure(figsize=(20,10))
sns.heatmap(dayHour)
sns.clustermap(dayHour)
dayMonth = df.groupby(by=['day','month']).count()['reason'].unstack()
dayMonth.head()
plt.figure(figsize=(12,6))
sns.heatmap(data=dayMonth)
sns.clustermap(data=dayMonth)
