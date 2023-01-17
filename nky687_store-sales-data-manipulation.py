import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
df=pd.read_csv('../input/rossmann-store-sales/train.csv',low_memory=False,parse_dates=['Date'])
df.head(5)
df.info()
dfg=df.groupby('Store').mean()

dfg.head()

dfg.reset_index(inplace=True)

dfg.head()
dfg.plot.scatter('Store','Sales',s=3,title='Avg sales per store')
# Store Sales across day of the week 

s_d=df.groupby(['Store','DayOfWeek'],as_index=False).mean()

for store in df['Store'].unique()[:5]:

    temp=s_d[s_d['Store']==store]

    plt.plot(temp.DayOfWeek,temp.Sales,label=f"Store {store}")

    plt.legend()

    plt.xlabel('Day of Week')

    plt.ylabel('Sales')

    plt.title('Sales on Day of Week')
# Continuous Grouping

df.groupby('Sales').mean().shape
df['Sales'].describe()
bins=[0,3000,5000,10000,20000,30000,50000]

df['SalesGrp']=pd.cut(df['Sales'],bins,include_lowest=True)

df.head()
df.groupby('SalesGrp',as_index=False).mean()

df.head()
df.groupby(['SalesGrp','Store']).DayOfWeek.value_counts().unstack(fill_value=0)
df.groupby(['Store','SalesGrp','DayOfWeek']).count()
plt.hist(df.Sales)
df=df[df.Open==1]

plt.hist(df.Sales)

df.shape
df.groupby(['Store','DayOfWeek']).agg({'Sales':['mean','min','max'],'Customers':'count'})

list(df)
def mc(x):

    return np.std(x)/np.sqrt(x.size)

df.groupby(['Store','DayOfWeek']).agg(

SalesMean=('Sales','mean'),

Salesuncer=('Sales',mc)

).reset_index().head()