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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

df.head()
df['Departments'] = df['title'].apply(lambda title: title.split(':')[1])

df.head()
df['Reason'].value_counts().head(1)
sns.countplot(x='Reason',data=df,palette='magma')

sns.despine(left=True)
df['Departments'].value_counts()
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df.head()
day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(day_map)
df.head()
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

sns.despine(left=True)

#To keep the legend out of the plot

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
sns.countplot(x='Month',data=df,hue='Reason',palette='magma')

sns.despine(left=True)

#To keep the legend out of the plot

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
Month_grouping = df.groupby('Month').count()

Month_grouping.head()
Month_grouping['twp'].plot()

plt.show()

#Here we will understand the curve or trend a little bit better than the bar plot
df['Date']=df['timeStamp'].apply(lambda t: t.date())

df.head()
#Importing plotly and cufflinks for creating interactive plots

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

import plotly.plotly as py

import plotly.graph_objs as go



# For Notebooks

init_notebook_mode(connected=True)



# For offline use

cf.go_offline()
df.groupby('Date').count()['lat'].iplot(kind='line')
df[df['Reason']=='EMS'].groupby('Date').count()['lat'].iplot(kind='line')
df[df['Reason']=='Fire'].groupby('Date').count()['lat'].iplot(kind='line')
df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].iplot(kind='line')
df.groupby(by=['Day of Week','Hour']).count()['Reason']
df_hour=df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

df_hour.head()
plt.figure(figsize=(15,7))

sns.heatmap(df_hour,cmap='magma',linecolor='white',linewidths=1)

plt.show()
sns.clustermap(df_hour,cmap='coolwarm',linecolor='white',linewidths=1)

plt.show()
df_month=df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

df_month.head()
plt.figure(figsize=(15,7))

sns.heatmap(df_month,cmap='magma',linecolor='white',linewidths=1)

plt.show()
sns.clustermap(df_month,cmap='inferno',linecolor='white',linewidths=1)

plt.show()