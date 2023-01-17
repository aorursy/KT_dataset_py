import numpy as np 

import pandas as pd 
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
df=pd.read_csv("../input/911csv/911.csv")

df.head()
#top 5 zipcode

df['zip'].value_counts().head(5)
#Top 5 Township

df['twp'].value_counts().head(5)
# ** Take a look at the 'title' column, how many unique title codes are there? **

df['title'].nunique()
df['Reason']=df['title'].apply(lambda title: title.split(':')[0])

df['Reason']
df['Reason'].value_counts()
sns.countplot(x='Reason',data=df,palette='viridis')
type(df['timeStamp'].iloc()[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['timeStamp']
time = df['timeStamp'].iloc[0]

time.hour
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
df.head()
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth=df.groupby('Month').count()

byMonth.head()
byMonth['lat'].plot()
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
df['Date']=df['timeStamp'].apply(lambda t: t.date())

df.head()
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
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')