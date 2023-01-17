# The course's name is Python for Data Science and Machine Learning Bootcamp

# Imports

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Import data

df = pd.read_csv("/kaggle/input/montcoalert/911.csv")
df.info()
df.head()
df['Reason'] = df['title'].apply(lambda s: s.split(':')[0])
df['Reason'].value_counts()
sns.countplot(df['Reason'],palette='viridis')
df['timeStamp'] = df['timeStamp'].apply(pd.to_datetime)

df['Hour'] = df['timeStamp'].apply(lambda timestamp: timestamp.hour)

df['Month'] = df['timeStamp'].apply(lambda timestamp: timestamp.month)

df['Day of Week Number'] = df['timeStamp'].apply(lambda timestamp: timestamp.dayofweek)

df['Day of Week Name'] = df['timeStamp'].apply(lambda timestamp: timestamp.weekday_name)
plt.figure(figsize = (15,4))

sns.countplot(df['Day of Week Number'],palette='viridis');

dmap = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

plt.xticks(range(0,7),[dmap[val] for val in range(0,7)]);
plt.figure(figsize = (15,4))

sns.countplot(df['Day of Week Number'],hue = df['Reason'],palette = 'Paired')

plt.xticks(range(0,7),[dmap[val] for val in range(0,7)]);

plt.legend(loc = 'center left', bbox_to_anchor=(1.0, 0.5));
plt.figure(figsize = (15,4))

sns.countplot('Month',data=df,hue='Reason');
by_month = df.groupby(by='Month').count()
by_month
by_month['lat'].plot()
sns.lmplot(data=by_month.reset_index(),x='Month',y='twp')
df['Date'] = df['timeStamp'].apply(lambda timestamp: timestamp.date())
df['Reason'].value_counts()
plt.figure(figsize = (15,10))

df[df['Reason'] == 'Traffic'].groupby('Date').count()['twp'].plot(legend=False,title='Traffic')
plt.figure(figsize = (15,10))

df[df['Reason'] == 'EMS'].groupby('Date').count()['twp'].plot(legend=False,title='EMS')
plt.figure(figsize=(15,10))

df[df["Reason"] == 'Fire'].groupby('Date').count()['timeStamp'].plot(title='Fire',legend='False')
new_df = df.groupby(['Hour','Day of Week Name']).count().unstack()['lat'].transpose()
plt.figure(figsize=(12,7))

sns.heatmap(data = new_df,cmap='viridis')
sns.clustermap(data=new_df,cmap = 'viridis')