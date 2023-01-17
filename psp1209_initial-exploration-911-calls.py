# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')



# Read data 

df=pd.read_csv("../input/911.csv",

    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],

    dtype={'lat':str,'lng':str,'desc':str,'zip':str,

                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 

     parse_dates=['timeStamp'],date_parser=dateparse)





# Set index

df.index = pd.DatetimeIndex(df.timeStamp)

df=df[(df.timeStamp >= "2016-01-01 00:00:00")]
df.head()
#Basic Info
df.info()
#The top 5 zip codes 
df['zip'].value_counts().head()
#Top 5 Townships
df['twp'].value_counts().head(5)
#Making Reason column, ex EMS: BACK PAINS/INJURY, Reason  = EMS
def newCol(rsn):

    d = rsn.split(':')

    return d[0]
df['Reason'] = list(map(newCol,df['title']))
df['Reason'].value_counts()
#EMS is the Reason most calls are made
# Reason plot
sns.countplot('Reason',data= df)
#Splitting date time into Hour, Month, Week
df['Hour'] = df['timeStamp'].apply(lambda t : t.hour)

df['Month'] = df['timeStamp'].apply(lambda t: t.month)

df['Week'] = df['timeStamp'].apply(lambda t:t.dayofweek)
#new Data

df.head()
#converting week to Monday - Sunday
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of the Week'] = df['Week'].map(dmap)
#Dropping column e

df =df.drop('e',axis=1)
#countplot of the Day of Week column with the hue Reason
sns.countplot('Day of the Week',data=df,hue='Reason')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#for month

sns.countplot('Month',data=df,hue='Reason')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#grouping by month

byMonth = df.groupby('Month').count()

byMonth.head()
#count of calls per month

byMonth['twp'].plot()
# a linear fit to the plot

sns.lmplot(x= 'Month',y='twp',data=byMonth.reset_index())
#new column date

df['Date'] = df['timeStamp'].apply(lambda t : t.date())

df['Date'].head()
#grouping by date and count of calls

d = df.groupby('Date').count()['twp'].plot()
#seperating plot based off of reason

df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot(figsize=(7,4))
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
#restructuring df , making hour the column

df1 = df.groupby(by = ['Day of the Week','Hour']).count()['Reason'].unstack()
plt.figure(figsize=(12,8))

sns.heatmap(df1,cmap='inferno')
#Month as the column

df2 = df.groupby(by=['Day of the Week','Month']).count()['twp'].unstack()
plt.figure(figsize=(12,8))

sns.heatmap(df2,cmap='inferno')