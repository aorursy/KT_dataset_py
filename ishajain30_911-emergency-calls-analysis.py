import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df= pd.read_csv('../input/911.csv')
df.info()

df.head()

df['zip'].value_counts().head(5)

df['twp'].value_counts().head(5)

df['title'].nunique()

df['Reason']=df['title'].apply(lambda title: title.split(':')[0])
df['Reason']

df['Reason'].value_counts()

sns.countplot(x='Reason', data=df)
type(df['timeStamp'].iloc[2])

df['timeStamp']=pd.to_datetime(df['timeStamp'])
df['Month']= df['timeStamp'].apply(lambda time: time.month)
df['Hour']= df['timeStamp'].apply(lambda time: time.hour)
df['day']= df['timeStamp'].apply(lambda time: time.dayofweek)
df['day'].unique()

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day']=df['day'].map(dmap)
df['day']


df.info()
sns.countplot(x='day', data=df, hue='Reason', palette='viridis')
plt.legend(bbox_to_anchor=(1.0,1.0))
sns.countplot(x='Month', data=df, hue='Reason', palette='viridis')
plt.legend(bbox_to_anchor=(1.2,1.0))
bymonth= df.groupby('Month').count()
bymonth.head()

bymonth['twp'].plot()
sns.lmplot('Month', 'twp', data=bymonth.reset_index())
df['Date']=df['timeStamp'].apply(lambda time: time.date())
df['Date']

df.groupby('Date').count()['twp'].plot()
plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()
df.info()
dayHour = df.groupby(by=['day','Hour']).count()['Reason'].unstack()
dayHour.head()

plt.figure(figsize=(8,6))
sns.heatmap(dayHour, cmap='viridis')
plt.tight_layout()

sns.clustermap(dayHour, figsize=(6,4), cmap='viridis')
plt.tight_layout()
datamonth = df.groupby(by=('day', 'Month')).count()['Reason'].unstack()
datamonth.head()
sns.heatmap(datamonth, cmap='viridis')
plt.figure(figsize=(6,8))
sns.clustermap(datamonth, cmap='viridis')