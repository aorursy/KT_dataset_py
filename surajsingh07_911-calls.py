# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Libraries for data analysis

import numpy as np

import pandas as pd



# Libraries for Visualization

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/911-calls/911.csv')
df.head()
df.info()
df.describe()
# Finding out which column has missing data and how much ?

df.isnull().sum()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['Reason'].value_counts()[:1]
sns.set_style('whitegrid')

sns.countplot(x='Reason', data=df)
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}



df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot(x='Day of Week', data=df, hue='Reason')

plt.legend(loc='best', bbox_to_anchor=(1.25,0.5))
sns.countplot(x='Month', data=df, hue='Reason')

plt.legend(bbox_to_anchor=(1.25,0.5))
byMonth = df.groupby('Month').count()
byMonth
# Could be any column

byMonth['twp'].plot()
sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())
df['Date'] = df['timeStamp'].apply(lambda time: time.date())
plt.figure(figsize=(10,5))

df.groupby('Date').count()['twp'].plot()

plt.tight_layout()
fig, axs = plt.subplots(nrows=3, figsize=(10,6))



df[df['Reason']=='EMS'].groupby('Date')['twp'].count().plot(ax=axs[0])

axs[0].set_title('EMS',fontsize=15)

df[df['Reason']=='Traffic'].groupby('Date')['twp'].count().plot(ax=axs[1])

axs[1].set_title('Traffic',fontsize=15)

df[df['Reason']=='Fire'].groupby('Date')['twp'].count().plot(ax=axs[2])

axs[2].set_title('Fire',fontsize=15)



plt.tight_layout()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')
plt.figure(figsize=(12,6))

sns.clustermap(dayHour, cmap='viridis')
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

dayMonth
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth, cmap='viridis')
plt.figure(figsize=(12,6))

sns.clustermap(dayMonth, cmap='viridis')