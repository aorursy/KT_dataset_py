import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

print(os.listdir("../input"))

# Reading the given data

df = pd.read_csv('../input/911.csv')

#displaying the head of the data

df.head()
#checking the data info

df.info()
df['twp'].value_counts().head(5)
df['zip'].value_counts().head(5)
df['title'].value_counts().head(5)
#number of unique reasons['title'] for calling 911

df['title'].nunique()
df['reasons']=df['title'].apply(lambda title: title.split(':')[0])

df['reasons'].value_counts()
sns.countplot(x=df['reasons'],data=df)

plt.show()
#checking the data type of the column timeStamps

type(df['timeStamp'].iloc[0])
#in order to further manipulate the data lets convert the timeStamp form str to DataTime objects

df['timeStamp']=pd.to_datetime(df['timeStamp'])
df['Hour']=df['timeStamp'].apply(lambda time: time.hour)

df['Month']=df['timeStamp'].apply(lambda time: time.month)

df['Day of Week']=df['timeStamp'].apply(lambda time: time.dayofweek)
df['Day of Week'].unique()
# creating the dictionary

dmap={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

# Changing the day of the week column to have proper strings

df['Day of Week']=df['Day of Week'].map(dmap)
#making sure that the changes took place

df['Day of Week'].unique()
sns.countplot(x=df['Day of Week'], data=df,hue='reasons')

plt.legend(bbox_to_anchor=(1.05,1),loc=(2),borderaxespad=0.)

plt.show()
sns.countplot(x=df['Month'], data=df,hue='reasons')

plt.legend(bbox_to_anchor=(1.05,1),loc=(2),borderaxespad=0.)

plt.show()
df['Date']=df['timeStamp'].apply(lambda t: t.date())
df.groupby('Date').count()['twp'].plot()

plt.tight_layout()
df[df['reasons']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()
df[df['reasons']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
df[df['reasons']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
dayHour=df.groupby(by=['Day of Week','Hour']).count()['reasons'].unstack()

dayHour.head()
sns.heatmap(dayHour,cmap='YlGnBu').set_title('HeatMap for dayHour')

plt.show()
sns.clustermap(dayHour,cmap='mako').fig.suptitle('ClusterMap for dayHour')

plt.show()
dayMonth=df.groupby(by=['Day of Week','Month']).count()['reasons'].unstack()

dayMonth.head()
sns.heatmap(dayMonth,cmap='mako').set_title('HeatMap for dayMonth')

plt.show()
sns.clustermap(dayMonth,cmap='mako').fig.suptitle('ClusterMap for dayMonth')

plt.show()