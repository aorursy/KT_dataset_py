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
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/911.csv") 
df.info()
df.head()
#top 5 zip codes



df['zip'].value_counts().head(5)
#top 5 townships 

df['twp'].value_counts().head(5)
#unique values in 'title' column







'''

or df['title'].nunique

'''

len(df['title'].unique())
#feature engineering



x = df['title'].iloc[0]
x.split(':')
x.split(':')[0]
#for entire title column



df['Reason'] = df['title'].apply(lambda title:title.split(':')[0])
#most common reason for the 911 call

df['Reason'].value_counts()
#count plot for 'Reason'

sns.countplot(x='Reason',data=df)

#datatype of timestamp column



df.info('all')
#datatype of timestamp column

type(df['timeStamp'].iloc[0])
#changing str to datetime object

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
time = df['timeStamp'].iloc[0]
time
time.minute
#making 3 different columns from timeStamp-> hour, month, day of week



df['hour'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.hour)
df['hour']
df['month'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.month)
df['month'].value_counts()
df['dayofweek'] = df['timeStamp'].apply(lambda timeStamp: timeStamp.dayofweek)
df['dayofweek']
df.head(5)
#map actual string names to day of week



dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}



df['dayofweek'] = df['dayofweek'].map(dmap)
df['dayofweek']
#count plot for 'day of week' and hue based on 'Reason' col



sns.countplot(x='dayofweek',data=df, hue='Reason')



#to relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0)
#count plot for 'month' and hue based on 'Reason' col

sns.countplot(x='month',data=df, hue='Reason')



#to relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0)
#dealing with missing months using groupby



bymonth = df.groupby('month').count()
bymonth.head()
#plotting line plot



bymonth['lat'].plot()
#linear fit model for No. of calls per month 



sns.lmplot(x='month', y='twp', data=bymonth.reset_index())
#new col containing date



df['date'] = df['timeStamp'].apply(lambda timeStamp: timeStamp.date())
df['date'].value_counts().head()
#groupby date



bydate = df.groupby('date').count()
bydate
#plot of count by date



bydate['lat'].plot()

plt.tight_layout()
#3 separate plots on the basis of reasons



df.groupby('date').count()['lat'].plot()

plt.tight_layout()
df[df['Reason']=='EMS'].groupby('date').count()['lat'].plot()

plt.title('EMS')

plt.tight_layout()
df[df['Reason']=='Fire'].groupby('date').count()['lat'].plot()

plt.title('Fire')

plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('date').count()['lat'].plot()

plt.title('Traffic')

plt.tight_layout()
#creating heatmaps



#grouping by 2 cols -> than using unstack to make one as col and one as row 



dayhour = df.groupby(by=['dayofweek','hour']).count()['Reason'].unstack()

#heatmap

plt.figure(figsize=(12,6))

sns.heatmap(dayhour)
#month and day of week



daymonth = df.groupby(by=['month','dayofweek']).count()['Reason'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(daymonth)
sns.clustermap(dayhour, cmap='coolwarm')
sns.clustermap(daymonth, cmap='coolwarm')