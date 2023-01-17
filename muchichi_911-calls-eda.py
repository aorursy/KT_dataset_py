# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/911.csv")

df.head()
df.columns # Get to know the data structure
df.info()
type(df['timeStamp'].iloc[0])
df['zip'].value_counts().head()
df['twp'].value_counts().head()
df['twp'].value_counts().tail()
df['title'].unique()[0]
x = df['title'].iloc[0]

print(x)

print(x.split(':')[0])
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

df['Reason Detail'] = df['title'].apply(lambda title: title.split(':')[1])

df['Reason Detail'].head()
df['Reason'].value_counts()
df['Reason Detail'].value_counts().head()
sns.countplot(x='Reason',data=df,palette ='viridis') # Countplot of 911 calls by Reason
print(df['timeStamp'].iloc[0])

print(type(df['timeStamp'].iloc[0]))
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
time = df['timeStamp'].iloc[0]

time.dayofweek
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}  # Map attribute with other values}
df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot(x='Day of Week',data=df,hue='Reason')

plt.legend(bbox_to_anchor=(1.05,1),loc=2, borderaxespad=0)
sns.countplot(x='Month',data=df,hue='Reason')

plt.legend(bbox_to_anchor=(1.05,1),loc=2, borderaxespad=0)
byMonth = df.groupby('Month').count()

byMonth.head(12)
byMonth['lat'].plot()
sns.lmplot(x='Month',y='twp',data = byMonth.reset_index())
t = df['timeStamp'].iloc[0].date()

t
df['Date'] = df['timeStamp'].apply(lambda time: time.date())
df.groupby('Date').count().head()
df.groupby('Date').count()['lat'].plot()

plt.tight_layout()
df[df['Reason']=='EMS'].groupby('Date').count()['lat'].plot()

plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].plot()

plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['lat'].plot()

plt.tight_layout()
#Heat map

df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

sns.heatmap(dayHour,cmap='viridis')

plt.figure(figsize=(12,6))
sns.clustermap(dayHour,cmap='viridis')
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

dayMonth.head()
sns.heatmap(dayMonth,cmap='viridis')
sns.clustermap(dayMonth,cmap='coolwarm')