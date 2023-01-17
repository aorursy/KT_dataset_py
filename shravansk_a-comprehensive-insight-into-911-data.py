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

df = pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().head(5) # Top five zip codes
df['twp'].value_counts().head(5) # Top five townships for 911 calls
# Create a separate column which contains reason why call was made : EMS, Traffic or Fire
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
df.head()
sns.countplot(df['Reason'])
emstype = df[df['Reason'] == 'EMS']['title']
emstype.value_counts().head(10)
firetype = df[df['Reason'] == 'Fire']['title']
firetype.value_counts().head(10)
traffictype = df[df['Reason'] == 'Traffic']['title']
traffictype.value_counts().head(10)
plt.figure(figsize=(8,6))
ax = sns.countplot( traffictype)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
#plt.set_xlabel('Fire calls type')
plt.tight_layout()
plt.show()
#timeStamp column contains a lot of information which can be extracted.
type(df['timeStamp'][0])
#timeStamp entries are a string. We have to convert it to datetime format

df['timeStamp']= pd.to_datetime(df['timeStamp'])
type(df['timeStamp'][0])
#Create three new columns which contain information of day, date, hour
df['Date']=df['timeStamp'].apply(lambda x: x.date())
df['Day'] =df['timeStamp'].apply(lambda x:x.dayofweek)
df['Hour'] =df['timeStamp'].apply(lambda x:x.hour)
df.head()


# Day is in number, we want to map it to Monday to Sunaday so we use .map() function
getday={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day']=df['Day'].map(getday)
df.head()
plt.figure(figsize=(12,6))
df.groupby('Date')
df2=df.groupby('Date').count()
df2['twp'].plot.line()
plt.figure(figsize=(12,6))
df3 = df[df['Reason'] == 'Traffic'].groupby('Date').count()
df3['twp'].plot.line()

plt.figure(figsize=(12,6))
df3 = df[df['Reason'] == 'Fire'].groupby('Date').count()
df3['twp'].plot.line()

plt.figure(figsize=(12,6))
df3 = df[df['Reason'] == 'EMS'].groupby('Date').count()
df3['twp'].plot.line()

#dayhour = df.groupby(by=['Reason','Day', 'Hour']).count()['lat'].unstack('Hour')
dayHour = df.groupby(by=['Day','Hour']).count()['Reason'].unstack()
dayHour.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayHour)
