import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Import the data into a dataframe
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
#check for any missing data
df.isnull().sum()
#The top 5 zip codes for 911 calls with the data we have
df['zip'].value_counts().head()
#The top 5 townships for 911 calls with the data we have
df['twp'].value_counts().head()
#Let's see how many unique title codes there are.
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda st: st.split(':')[0])
df.head()
df['Reason'].value_counts()
#let's visualize the above result to visually compare these numbers
sns.countplot(x='Reason',data=df)
#checking the timestamp column datatype
type(df['timeStamp'].iloc[0])
#convert them to DateTime objects
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek).map(dmap)
#There is a time.weekday_name attribute that could have produced an equivalent solution, but I wanted to practice mapping a dictionary
#check to see our new dataframe
df.head()
sns.countplot(x='Day of Week', data = df, hue = 'Reason', palette = 'Set2')
plt.legend(loc='lower left',bbox_to_anchor=(1.0,0.5))
sns.countplot(x='Month', data = df, hue = 'Reason', palette = 'Set2')
plt.legend(loc='lower left',bbox_to_anchor=(1.0,0.5))
df.groupby('Month').count()
#let's turn this into a graph to better understand calling trends per month
df.groupby('Month').count().plot.line(use_index = True,y = 'title',legend = None)
plt.ylabel('count')
sns.lmplot(x='Month',y = 'title', data = df.groupby('Month').count().reset_index())
plt.ylabel('count')
#let's use the timestamp information to create a new column
df['Date'] = df['timeStamp'].apply(lambda ts: ts.date())
df.head()
df.groupby('Date').count().plot.line(use_index = True, y = 'title', figsize= (15,2), legend = None)
plt.ylabel('count')
df['Date'] = pd.to_datetime(df['Date'])
df.groupby(df[df['Date'].dt.year>=2018]['Date']).count().plot.line(use_index = True, y = 'title', legend = None)
plt.ylabel('count')
df.groupby(df[(df['Date'].dt.year>= 2018) & (df['Date'].dt.month==3)]['Date']).count()
#Checking the reasons to see if it's distributed according to the entire dataset.
df[df['Date']=='2018-03-02']['Reason'].value_counts()

sns.countplot(x='Reason',data=df[df['Date']=='2018-03-02'])
#reusing the same code from before
df.groupby(df[(df['Date'].dt.year>= 2018) & (df['Date'].dt.month==11)]['Date']).count()
sns.countplot(x='Reason',data=df[df['Date']=='2018-11-15'])
#unstacking and viewing the groupby table so we can find out how to select our data.
df.groupby(['Date','Reason']).count().unstack()
#Traffic
df.groupby(['Date','Reason']).count()['title'].unstack().plot.line(use_index = True, y = 'Traffic', figsize= (15,2), legend = None)
plt.title('Traffic')
plt.ylabel('count')
#EMS
df.groupby(['Date','Reason']).count()['title'].unstack().plot.line(use_index = True, y = 'EMS', figsize= (15,2), legend = None)
plt.title('EMS')
plt.ylabel('count')
#Fire
df.groupby(['Date','Reason']).count()['title'].unstack().plot.line(use_index = True, y = 'Fire', figsize= (15,2), legend = None)
plt.title('Fire')
plt.ylabel('count')
#First need to change the dataframe to a pivot table with days of week and hours in day
dfht = df.groupby(['Day of Week','Hour']).count().unstack()['title']
dfht
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(dfht, cmap='coolwarm',ax = ax)
sns.clustermap(dfht, cmap = 'coolwarm', figsize = (12,10))
#Creating the dataframe we'll use
dfmt = df.groupby(['Day of Week','Month']).count().unstack()['title']
dfmt
#Heatmap
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(dfmt, cmap='coolwarm', ax = ax)
#let's make a cluster map of the same information
sns.clustermap(dfmt, cmap = 'coolwarm', figsize = (12,10))
