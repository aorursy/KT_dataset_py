# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

%matplotlib inline
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().head()
df['twp'].value_counts().head()
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['Reason'].value_counts()
sns.set_style("darkgrid")

sns.countplot(x='Reason',data=df, palette = 'viridis')
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

type(df['timeStamp'].iloc[0])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot(x='Day of Week', data=df, hue='Reason', palette='viridis')

#to relocate the legend

plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
sns.countplot(x='Month', data=df, hue='Reason', palette='viridis')

#to relocate the legend

plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
byMonth = df.groupby('Month').count()

byMonth.head()
byMonth['lat'].plot()
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
df['Date'] = df['timeStamp'].apply(lambda t:t.date())
df.groupby('Date').count()['lat'].plot()

plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].plot()

plt.title('Traffic')

plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['lat'].plot()

plt.title('Fire')

plt.tight_layout()
df[df['Reason']=='EMS'].groupby('Date').count()['lat'].plot()

plt.title('EMS')

plt.tight_layout()
dayHour= df.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour, cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
dayMonth = df.groupby(by=['Day of Week', 'Month']).count()['Reason'].unstack()

dayMonth.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth, cmap='viridis')
sns.clustermap(dayMonth,cmap='viridis')