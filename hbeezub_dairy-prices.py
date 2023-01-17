import csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import warnings 

warnings.filterwarnings("ignore")

import seaborn as sns

import datetime as dt
data = pd.read_csv('../input/Datamart-Export_DY_WK100-500 Pound Barrel Cheddar Cheese Prices, Sales, and Moisture Content_20170829_122601.csv')
data.shape
data.head(10)
type(data)
data.dtypes
data.isnull().any()
data.describe()
data['mo_day']=data['Date'].astype(str)+'-1999'

data.head()
data['Week Ending Date'] = pd.to_datetime(data['Week Ending Date'])



data['year'] = data['Week Ending Date'].dt.year

data.head(5)

data.tail(5)
data['mo_day'] = pd.to_datetime(data['mo_day'])

data['month'],data['day'] = data['mo_day'].dt.month, data['mo_day'].dt.day 

data.head(10)
data.dtypes
data['Date_yr'] = data['year'].map(str)+'-' + data['month'].map(str) +'-'+ data['day'].map(str)     

data.head(5)


data['Date_yr'] = pd.to_datetime(data['Date_yr'])

data['Week Ending Date'] = pd.to_datetime(data['Week Ending Date'])

data.head(5)

data.dtypes
data['Sales'] = data['Sales'].str.replace(',', '')

data['Sales'] = pd.to_numeric(data['Sales'])          

data.dtypes
data['age']=(data['Week Ending Date']-data['Date_yr']).dt.days

data.head(10)
data.dtypes
data.iloc[140:170,]
data['age'][data['age'] < 0] = 365+data['age']

data.iloc[140:170,]

#finally 3 hours!
df1=data

del df1['month']

del df1['day']

df1.head(5)

years = df1.set_index("year")

years.head()

years.tail()
#2017 data

df2017=df1.iloc[0:165,]  #line 165 is 2016 so we need 1 more than line 164 ie 165

#df2017.head()

df2017.tail()
df2016=df1.iloc[166:430,]  #line 430 is 2015 so we need 1 more than line 429 ie 430

#df2016.head()

df2016.tail(7)
df2015=df1.iloc[430:690,]  #line 690 is 2014 so we need 1 more than line 690 ie 691

#df2015.head()

df2015.tail(7)
df2014=df1.iloc[690:950,]  #line 950 is 2013

#df2014.head()

df2014.tail(7)
test = df2017

test['Week Ending Date'] = pd.to_datetime(test['Week Ending Date'])

test = test.set_index('Week Ending Date')

title=('sales per week')

ax = test.plot()

plt.show()
fig, ax = plt.subplots()

for name, group in df2017.groupby('age'):

    group.plot('Week Ending Date', y='Sales', ax=ax, label=name)

    ax.set_title('Sales 2017')

plt.show()
fig, ax = plt.subplots()

for name, group in df2017.groupby('age'):

    group.plot('Week Ending Date', y='Moisture Content', ax=ax, label=name)

    ax.set_title('Moisture Content 2017')

plt.show()
fig, ax = plt.subplots()

for name, group in df2017.groupby('age'):

    group.plot('Week Ending Date', y='Weighted Price', ax=ax, label=name)

    ax.set_title('Weighted Price per week 2017')

plt.show()
fig, ax = plt.subplots()

for name, group in df2016.groupby('age'):

    group.plot('Week Ending Date', y='Weighted Price', ax=ax, label=name)

    ax.set_title('Weighted Price per week 2016')

plt.show()
fig, ax = plt.subplots()

for name, group in df2015.groupby('age'):

    group.plot('Week Ending Date', y='Sales', ax=ax, label=name, figsize=(12,4))

    ax.set_title('Sales 2015')

plt.show()
fig, ax = plt.subplots()

for name, group in df2015.groupby('age'):

    group.plot('Week Ending Date', y='Weighted Price', ax=ax, label=name, figsize=(12,4))

    ax.set_title('Weighted Price per week 2015')

plt.show()
fig, ax = plt.subplots()

for name, group in df2014.groupby('age'):

    group.plot('Week Ending Date', y='Sales', ax=ax, label=name, figsize=(12,4))

    ax.set_title('Sales 2014')

plt.show()
fig, ax = plt.subplots()

for name, group in df2014.groupby('age'):

    group.plot('Week Ending Date', y='Weighted Price', ax=ax, label=name, figsize=(12,4))

    ax.set_title('Weighted Price per week 2014')

plt.show()
df1.set_index(keys=['age'], drop=False,inplace=True)

ages=df1['age'].unique().tolist()

df1_14 = df1.loc[df1.age==14]               

df1_14.head(5)

#df1_14.tail(5)
df1_14.shape

fig, ax = plt.subplots()

for name, group in df1_14.groupby('year'):

    group.plot('Date', y='Weighted Price', ax=ax, label=name,figsize=(12,4))

    ax.set_title('Weighted price 14 day age cheddar')

plt.show()
fig, ax = plt.subplots()

for name, group in df1_14.groupby('year'):

    group.plot('Date', y='Sales', ax=ax, label=name,figsize=(12,4))

    ax.set_title('Sales 14 day age cheddar')

plt.show()
df1.set_index(keys=['age'], drop=False,inplace=True)

ages=df1['age'].unique().tolist()

df1_28 = df1.loc[df1.age==28]        

df1_28.head(5)

#df1_28.tail(5)
fig, ax = plt.subplots()

for name, group in df1_28.groupby('year'):

    group.plot('Date', y='Weighted Price', ax=ax, label=name,figsize=(12,4))

    ax.set_title('Weighted price 28 day age cheddar')

plt.show()
fig, ax = plt.subplots()

for name, group in df1_28.groupby('year'):

    group.plot('Date', y='Sales', ax=ax, label=name,figsize=(12,4))

    ax.set_title('Salese 28 day age cheddar')

plt.show()
df1_14.set_index(keys=['year'], drop=False,inplace=True)

years_14=df1_14['year'].unique().tolist()

df1_14_2013_2015 = df1_14.loc[(df1_14.year>=2013) & (df1_14.year<=2015)]

df1_14_2013_2015.head(5)

df1_14_2013_2015.tail(5)
fig, ax = plt.subplots()

for name, group in df1_14_2013_2015.groupby('year'):

    group.plot('Date', y='Weighted Price', ax=ax, label=name,figsize=(12,4))

    ax.set_title('Weighted price 2013 vs 2015')

plt.show()

bar = df1_14_2013_2015.groupby("year").sum().plot(kind='bar', width=1.5)

bar_width = 0.4 

bar.set_xlabel("year")

bar.set_ylabel("Sales")

#plt.legend()

#plt.legend.remove()

plt.legend().set_visible(False)

plt.title('Total Sales')

plt.show()