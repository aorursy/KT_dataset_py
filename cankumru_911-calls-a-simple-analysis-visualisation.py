import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as plt

%matplotlib inline
df = pd.read_csv('../input/montcoalert/911.csv')
df.info()
df.head()
df['title'].unique()
df['Category']=df['title'].apply(lambda x: x.split(':')[0])

df['Detail']=df['title'].apply(lambda x: x.split(':')[1])

df=df.drop('title',1)
df['Category'].unique()
df['Detail'].unique()
type(df['timeStamp'].iloc[0])
df['timeStamp']=pd.to_datetime(df['timeStamp'])
df['Month']=df['timeStamp'].apply(lambda x: x.month)

df['Day']=df['timeStamp'].apply(lambda x: x.dayofweek)

df['Hour']=df['timeStamp'].apply(lambda x: x.hour)

df['Year']=df['timeStamp'].apply(lambda x: x.year)
df.head()
days = {0: 'Mon', 1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day']=df['Day'].map(days)

months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

df['Month']=df['Month'].map(months)
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['Day'].value_counts()
df['Month'].value_counts()
df['Category'].value_counts()
df['Detail'].value_counts().head(5)
byYear = df.groupby('Year').count()

byYear['twp'].plot()
sns.countplot(x='Year',hue='Category',data=df)
sns.countplot(x='Day',hue='Category',data=df,order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
sns.countplot(x='Month',hue='Category',data=df,order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
dH = df.groupby(by=['Day','Hour']).count()['Category'].unstack()

dH.head()
sns.heatmap(dH)
sns.clustermap(dH)
dM = df.groupby(by=['Day','Month']).count()['Category'].unstack()

dM.head()
sns.heatmap(dM)
sns.clustermap(dM)
cH = df.groupby(by=['Category','Hour']).count()['Detail'].unstack()

cH.head()
sns.heatmap(cH)
sns.clustermap(cH)
cC=df.groupby(by=['Category','Day']).count()['Detail'].unstack()

cC.head()
sns.heatmap(cC)
sns.clustermap(cC)