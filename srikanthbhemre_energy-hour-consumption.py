from IPython.display import HTML

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import math

import scipy.stats as ss

import warnings 

import pandas_profiling

warnings.simplefilter('ignore')
df=pd.read_csv('../input/PJME_hourly.csv', parse_dates=['Datetime'])
df.shape
df.head()
df.describe()
df.info()
pandas_profiling.ProfileReport(df)
df.PJME_MW.plot(kind='hist')
df.PJME_MW.plot()
df['timeStamp'] = pd.to_datetime(df['Datetime'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Date'] = df['timeStamp'].apply(lambda t: t.day)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Year'] = df['timeStamp'].apply(lambda t: t.year)

df['dayname']=df['Datetime'].dt.day_name()
print('unique years',df['Year'].unique())

print('unique months',df['Month'].unique())
plt.figure(figsize=(15,5))

plt.title(' Total sell throughout 2002-2018')

sns.countplot(x='Year', data=df, color='lightblue');
plt.figure(figsize=(10,5))

plt.title(' Total PJME_MW on each month throughout 2002-2018')

plt.ylabel('PJME_MW')

df.groupby('Month').PJME_MW.sum().plot(kind='bar',color='lightblue')
data_2002 = df[df["Year"] == 2002]

data_2003 = df[df["Year"] == 2003]

data_2004 = df[df["Year"] == 2004]

data_2005 = df[df["Year"] == 2005]

data_2006 = df[df["Year"] == 2006]

data_2007 = df[df["Year"] == 2007]

data_2008 = df[df["Year"] == 2008]

data_2009 = df[df["Year"] == 2009]

data_2010 = df[df["Year"] == 2010]

data_2011 = df[df["Year"] == 2011]

data_2012 = df[df["Year"] == 2012]

data_2013 = df[df["Year"] == 2013]

data_2014 = df[df["Year"] == 2014]

data_2015 = df[df["Year"] == 2015]

data_2016 = df[df["Year"] == 2016]

data_2017 = df[df["Year"] == 2017]

data_2018 = df[df["Year"] == 2018]





plt.figure(figsize=(10,7))

plt.title(' PJME_MW on each month throughout 2002')

plt.ylabel('PJME_MW')

data_2002.groupby('Month').PJME_MW.sum().plot(kind='bar',color='lightblue')
plt.figure(figsize=(10,7))

plt.title(' PJME_MW on each month throughout 2003')

plt.ylabel('PJME_MW')

data_2003.groupby('Month').PJME_MW.sum().plot(kind='bar')
plt.figure(figsize=(10,7))

plt.title(' PJME_MW on each month throughout 2004')

plt.ylabel('PJME_MW')

data_2004.groupby('Month').PJME_MW.sum().plot(kind='bar')
df.tail()
# df.drop(['Datetime','timeStamp'], axis=1, inplace=True)

df.index
df.tail()
df.Date.unique()
sns.scatterplot(x='Hour',y='PJME_MW',data=df)
sns.scatterplot(x='Date',y='PJME_MW',data=df)
sns.scatterplot(x='Month',y='PJME_MW',data=df)
sns.scatterplot(x='Year',y='PJME_MW',data=df)


sns.heatmap(df.corr(),annot=True)
sns.lineplot(x='Year', y='PJME_MW', data=df)
sns.lineplot(x='Month', y='PJME_MW', data=df)
features = ['Hour', 'Day of Week']
from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split
X = df[features]

y = df['PJME_MW']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 125)

linreg = LinearRegression()

linreg.fit(X_train,y_train)

y_pred = linreg.predict(X_test)

from sklearn import metrics

np.sqrt(metrics.mean_squared_error(y_test,y_pred))