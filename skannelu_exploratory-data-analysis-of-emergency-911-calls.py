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
#Import numpy, pandas, visualization libraries and set %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# Read in the csv file as a dataframe called df
df = pd.read_csv('../input/911.csv')
#Exploratory Data Analysis---------------
df.info()
df.head(10)
# Top 10 zipcodes for 911 calls
df['zip'].value_counts().head(10)
#Top 10 townships for 911 calls
df['twp'].value_counts().head(10)
#Top 10 Most occurred emergencies for 911 calls
df['title'].value_counts().head(10)
# Most of the Durations for the occured emergencies 911 calls
df['timeStamp'].value_counts().head(10)
#Creating a new column 'Reason'
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
#Most occured types of emergency 911 calls
df['Reason'].value_counts()
#Countplot of 911 calls by 'Reason' column
sns.countplot(x='Reason',data=df,palette='viridis')
#Converting timeStamp column from str type to DateTime objects
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
#Creating 3 columns from timeStamp column
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
# Months where most of emergency 911 calls occured
df['Hour'].value_counts()
# Day where most of emergency 911 calls occured 
df['Month'].value_counts()
# Hour where most of emergency 911 calls occured 
df['Day of Week'].value_counts()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')
#Re-locating the legend
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
#Grouping the dataframe by Month column
byMonth = df.groupby('Month').count()
byMonth.head()
#Simple plot to indicate the count of calls per month
#byMonth['lat'].plot()
sns.countplot(x='Month', data=df,palette='viridis') 
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
t = df['timeStamp'].iloc[0]
df['Date'] = df['timeStamp'].apply(lambda t:t.date())
#byDate = df.groupby('Date').count()
#byDate['lat'].plot()
# Separate plots representing type of emergency for the 911 calls
#df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].plot()
#plt.title('Traffic')
#plt.tight_layout()
#df[df['Reason']=='Fire'].groupby('Date').count()['lat'].plot()
#plt.title('Fire')
#plt.tight_layout()
#df[df['Reason']=='EMS'].groupby('Date').count()['lat'].plot()
#plt.title('EMS')
#plt.tight_layout()

#Matrix form sing unstack() method
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
#Heatmap
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')
#Clustermap
sns.clustermap(dayHour,cmap='coolwarm')

dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

#Heatmap for the data with Month as the column
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')
#Clustermap for the data with Month as the column
sns.clustermap(dayMonth,cmap='coolwarm')


