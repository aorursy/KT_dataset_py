import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

datapath = '../input/m5-forecasting-accuracy'



# import data files

calendar = pd.read_csv(f'{datapath}/calendar.csv', parse_dates=['date'])

sales_train_validation = pd.read_csv(f'{datapath}/sales_train_validation.csv')

# tags = 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'

# data = 'd_1' -> 'd_1913'

tags, data = sales_train_validation.iloc[:, :6], sales_train_validation.iloc[:, 6:]

data.columns = calendar.loc[:data.columns.size-1,'date']

df = data.sum(axis=0)

df.plot(figsize=(15,4))

data.columns = calendar.loc[:data.columns.size-1,'wday']

data.transpose().groupby('wday').mean().transpose().sum().plot()

data.columns = calendar.date.dt.day.loc[:data.columns.size-1]

data.transpose().groupby('date', sort=True).mean().transpose().sum().plot(figsize=(15,4))
calendar['weekno'] = ((calendar.date.dt.day-1)/7).astype(int) # zero based: 0 = week 1

dayno = 0

daynos = []

for index, row in calendar.iterrows():

    if dayno == 0:

        dayno = row.wday+row.weekno*7

    if (row.wday==1) & (row.weekno==0): # first Saturday

        dayno = 1

    daynos.append(dayno)

    dayno = dayno+1



calendar['dayno'] = daynos

data.columns = calendar.loc[:data.columns.size-1,'dayno']

data.transpose().groupby('dayno', sort=True).mean().transpose().sum().plot(figsize=(15,4))
df = calendar.groupby('dayno', sort=True)['date'].count().plot(figsize=(15,4))