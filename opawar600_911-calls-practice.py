# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.  
df = pd.read_csv("../input/911.csv")

df.head()
df.info()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
x = df['title'].iloc[0]

x.split(":")[0]
df['Reason'] = df['title'].apply(lambda title:title.split(':')[0])

df['Reason'].head(3)
df['Reason'].value_counts()
sns.countplot(df['Reason'])
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
time = df['timeStamp'].iloc[0]

time.hour
df['Hour'] = df['timeStamp'].apply(lambda time : time.hour)

df.tail()
df['Month'] = df['timeStamp'].apply(lambda time : time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time : time.dayofweek)
dic = { 0: 'Mon', 1:'Tue' , 2:'Wed' , 3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df["Day of Week"].map(dic)

df.head()
sns.countplot(df['Day of Week'],hue=df['Reason'])

plt.legend(bbox_to_anchor=(1.05,1), loc=2 , borderaxespad = 0.)
sns.countplot(df['Month'],hue=df['Reason'])

plt.legend(bbox_to_anchor=(1.05,1), loc=2 , borderaxespad = 0.)
df['Date'] = df['timeStamp'].apply(lambda t:t.date())

df.head()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour)
sns.clustermap(dayHour)