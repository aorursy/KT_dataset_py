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
#reading the 911 calls data 



df = pd.read_csv('../input/911.csv')
#info check of the DF



df.info()
#checking the head of the DF



df.head()
#top 5 zip-codes for 911 calls



df['zip'].value_counts().nlargest(5)
#top 5 townships for 911 calls



df['twp'].value_counts().nlargest(5)
#unique title in the df



df['title'].nunique()
#categorizing the title in a reason column



df['reasons'] = df['title'].apply(lambda x: x.split(':')[0])
#count of different reasons for 911 calls



df['reasons'].value_counts()
#imports for plotting



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#a count plot for reasons 



sns.countplot(df['reasons'])
#checking data type of timestamp



type(df['timeStamp'][0])
#converting string timestamp to time stamp object



df['timeStamp'] = pd.to_datetime(df['timeStamp'])
#checking a value after conversion



df['timeStamp'].iloc[0]
#creating columns for hour, month and week day from time stamp



df['hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['month'] = df['timeStamp'].apply(lambda x: x.month)

df['weekday'] = df['timeStamp'].apply(lambda x: x.dayofweek)
#dictionary for mapping weekday numbers



dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
#mapping the weekday numbers to day name



df['weekday'] = df['weekday'].map(dmap)
#using seaborn to create countplot of the weekday column with the hue based off of the Reason column



plt.subplots(figsize=(12,6))

sns.countplot(x='weekday', data=df, hue='reasons', palette='viridis')



#relocating the legends

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
#using seaborn to create countplot of the month column with the hue based off of the Reason column



plt.subplots(figsize=(12,6))

sns.countplot(x='month', data=df, hue='reasons', palette='viridis')



#relocating the legends

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#grouping the data by month



grpByMonth = df.groupby('month').count()



grpByMonth.head()
#a simple line plot for the grpByMonth DF



grpByMonth['twp'].plot()
#linear fit on the number of calls per month



sns.lmplot(x='month', y='twp', data=grpByMonth.reset_index())
#creating a new column for date. Date to be extracted from timeStamp column



df['date'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.date())
#grouping 911 calls by date



grpByDate = df.groupby('date').count()

grpByDate.head()
#plot on the 911 calls grouping by date



plt.subplots(figsize=(12,6))

grpByDate['twp'].plot()

plt.tight_layout()
#plot for 911 calls grouped by date and EMS Reason

plt.subplots(figsize=(12,6))

df[df['reasons']=='EMS'].groupby('date').count()['twp'].plot()

plt.tight_layout()
#plot for 911 calls grouped by date and Fire Reason



plt.subplots(figsize=(12,6))

df[df['reasons']=='Fire'].groupby('date').count()['twp'].plot()

plt.tight_layout()
#plot for 911 calls grouped by date and Traffic Reason



plt.subplots(figsize=(12,6))

df[df['reasons']=='Traffic'].groupby('date').count()['twp'].plot()

plt.tight_layout()
#restructuring the dataframe so that the columns become the Hours and the Index becomes the Day of the Week



dayHour = df.groupby(by=['weekday', 'hour']).count()['reasons'].unstack()

dayHour.head()
#heatmap for the dayHour DF



plt.subplots(figsize=(12,6))

sns.heatmap(dayHour, cmap='viridis')
#clustermap for the dayHour DF



sns.clustermap(dayHour, cmap='viridis')
#restructuring the dataframe so that the columns become the Months and the Index becomes the Day of the Week



dayMonth = df.groupby(by=['weekday', 'month']).count()['reasons'].unstack()

dayMonth.head()
#heatmap for the dayMonth DF



plt.subplots(figsize=(12,6))

sns.heatmap(dayMonth, cmap='viridis')
#clustermap for the dayHour DF



sns.clustermap(dayMonth, cmap='viridis')