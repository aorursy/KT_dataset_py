import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#import os
#print(os.listdir("../input"))
data = pd.read_csv('../input/911.csv')
data.head()
data.info()
data['zip'].value_counts().head()
data['twp'].value_counts().head()
data['title'].nunique()
data['Reason'] = data['title'].apply(lambda title: title.split(':')[0])
data['Reason'].head()
data['Reason'].value_counts()
sns.countplot(x='Reason',data=data)
type(data['timeStamp'].iloc[0])
data['timeStamp'] = pd.to_datetime(data['timeStamp'])
type(data['timeStamp'].iloc[0])
data['Hour'] = data['timeStamp'].apply(lambda time: time.hour)
data['Month'] = data['timeStamp'].apply(lambda time: time.month)
data['Day of Week'] = data['timeStamp'].apply(lambda time: time.dayofweek)
data.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
data['Day of Week'] = data['Day of Week'].map(dmap)
data.head()
sns.countplot(x='Day of Week', data =data, hue='Reason')
# Relocation of the legends outside
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
sns.countplot(x='Month', data =data, hue='Reason')
# Relocation of the legends outside
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
data['Date'] = data['timeStamp'].apply(lambda t: t.date())
data.head()
data.groupby('Date').count().head()
data.groupby('Date').count()['lat'].plot()
plt.tight_layout()
data[data['Reason']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.title("Traffic")
plt.tight_layout()
data[data['Reason']=='Fire'].groupby('Date').count()['lat'].plot()
plt.title("Fire")
plt.tight_layout()
data[data['Reason']=='EMS'].groupby('Date').count()['lat'].plot()
plt.title("EMS")
plt.tight_layout()
dayHour = data.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head(2)
sns.heatmap(dayHour,cmap='coolwarm')
sns.clustermap(dayHour,cmap='coolwarm')
dayMonth = data.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()
sns.heatmap(dayMonth,cmap='coolwarm')
sns.clustermap(dayMonth,cmap='coolwarm')
