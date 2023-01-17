import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/911calls-historic-data/911.csv') #importing our data set
df.head() #checking the head of our data
df['zip'].value_counts().head(5) #top 5 zip codes in our data
df['twp'].value_counts().head(5) #top 5 townships in our data
df['title'].nunique() #looking at the unique title codes
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0]) #refining our titles column
df['Reason'].value_counts() #checking for common reasons for 911 calls
sns.countplot(x='Reason',data=df,palette='inferno')
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='inferno')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='Month',data=df,hue='Reason',palette='inferno')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth = df.groupby('Month').count()

byMonth.head()
byMonth['twp'].plot() #this plot helped us visualise the above months section by filling the mmissing data
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index()) #we need to reset our index for this plot
df['Date']=df['timeStamp'].apply(lambda t: t.date())
plt.figure(figsize = (8,6))

df.groupby('Date').count()['twp'].plot()

plt.tight_layout()
plt.figure(figsize = (8,6))

df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()
plt.figure(figsize = (8,6))

df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
plt.figure(figsize = (8,6))

df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='inferno')
sns.clustermap(dayHour,cmap='inferno')
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

dayMonth.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='inferno')
sns.clustermap(dayMonth,cmap='inferno')