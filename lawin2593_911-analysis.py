import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/911.csv')
df.info()
df.head(n=3)
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda title : title.split(':')[0])
df['Reason'].value_counts()
import seaborn as sns
sns.countplot(x='Reason',data = df)
type(df['timeStamp'].iloc[0])
df['timeStamp']=pd.to_datetime(df['timeStamp'])
df['Hour']= df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time : time.month)
df['Day of the Week'] = df['timeStamp'].apply(lambda time : time.dayofweek)

dmap = {0:'Mon' ,1:'Tue', 2:'Wed', 3:'Thur', 4:'Fri',5:'Sat',6:'Sun'}
df['Day of the Week']=df['Day of the Week'].map(dmap)
sns.countplot('Day of the Week',data = df ,hue = 'Reason')
# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)

sns.countplot(x='Month',data=df,hue ='Reason')
#anchor legend
plt.legend(bbox_to_anchor = (1.05,1),loc=2,borderaxespad =0)
month_grp = df.groupby('Month').count()
month_grp.head(5)
month_grp['twp'].plot()
month_grp.reset_index()
sns.lmplot(x='Month',y='twp',data=month_grp.reset_index())
df['Date']=df['timeStamp'].apply(lambda x: x.date())
plt.figure(figsize=(12,4))
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.figure(figsize=(12,4))
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
plt.figure(figsize=(12,4))
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()
plt.figure(figsize=(12,4))
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()
dayHour = df.groupby(by=['Day of the Week','Hour']).count()['Reason'].unstack()
dayHour.head()
plt.figure(figsize =(10,6))
sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
monthhour = df.groupby(by=['Day of the Week','Month']).count()['Reason'].unstack()
monthhour.head()
plt.figure(figsize = (15,7))
sns.heatmap(monthhour,cmap = 'viridis')
plt.figure(figsize = (15,7))
sns.clustermap(monthhour,cmap = 'viridis')