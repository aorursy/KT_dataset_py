import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))


df=pd.read_csv('../input/911.csv')
df.isnull().sum()
df['title'].head(20)
df['Type'] = df['title'].apply(lambda x: x.split(':')[0])
df['Cause'] = df['title'].apply(lambda x: x.split(':')[1])
df

df['Type'].value_counts()
sns.countplot(x='Type',data=df,order=df['Type'].value_counts().index,palette='pastel')
df.groupby(['Type'])['Type'].size().sort_values(ascending=True).plot(kind='pie',autopct='%1.1f%%')
sns.countplot(x='zip',data=df,order=df['zip'].value_counts().head(5).index,palette='muted')
sns.countplot(y="Cause",data=df,order=df['Cause'].value_counts().head(5).index,palette='dark')
sns.countplot(y="twp",data=df,order=df['twp'].value_counts().head(5).index,color='brown')
#Splitting the timestamp into day,month and year columns
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day'] = df['timeStamp'].apply(lambda x: x.dayofweek)
df['quarter']=df['timeStamp'].dt.quarter
df['year']=df['timeStamp'].dt.year
df
d = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day'] = df['Day'].map(d)
df
byHour = df.groupby('Hour').count()
byHour['twp'].plot()
byMonth = df.groupby('Month').count()
byMonth['twp'].plot()
sns.countplot(df['quarter'])
sns.countplot(df['year'])
df[df['title'].str.contains("EMS")].groupby('title').size().sort_values(ascending=False).head().plot(kind='pie',autopct='%1.1f%%')

sns.countplot(x = 'Day', data = df, hue='Type', palette = 'rocket').legend(bbox_to_anchor=(1.05, 1), loc='center left')
sns.countplot(x = 'Month', data = df, hue='Type', palette = 'pastel').legend(bbox_to_anchor=(1.05, 1), loc='center left')
DayHour = df.groupby(by=['Day','Hour']).count()['Type'].unstack()
DayHour.head()

plt.figure(figsize=(12,6))
sns.heatmap(DayHour, cmap='viridis')

