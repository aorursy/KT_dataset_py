import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from statistics import mode, mean

import copy
data = pd.read_csv("/kaggle/input/vegetablepricetomato/Price of Tomato Karnataka(2016-2018).csv")
data.isnull().sum()
data.dropna(subset=['Variety'], inplace=True)

data.fillna(method='ffill', inplace=True)
data.isnull().sum()
data['Arrival Date'] = pd.to_datetime(data['Arrival Date'])

df_gb = data.groupby(['Arrival Date']).sum()

df_gb['periods']=df_gb.index.to_period("M")

df_p = df_gb.groupby(['periods']).sum()

print(df_p.head(8))
fig, ax = plt.subplots(figsize=(16,5))

ax.bar(df_gb.index, df_gb['Arrivals (Tonnes)'])

ax.set_ylabel('Arrivals (Tonnes)')

plt.gcf().autofmt_xdate()
data['Modal Price(Rs./Quintal)'] = data['Modal Price(Rs./Quintal)'].astype('int')
fig, ax = plt.subplots(figsize=(9,9))

ax.scatter(data['Arrivals (Tonnes)'], data['Market'], marker='*');

ax.set(xlabel='Arrivals (Tonnes) in market', ylabel='Market', title='The volume of supply of tomatoes to the market');
data['revenue'] = data['Arrivals (Tonnes)']*data['Modal Price(Rs./Quintal)']*10
dt = pd.pivot_table(data, values='revenue', index=['Market'], aggfunc=np.sum)
fig, ax = plt.subplots(figsize=(18, 4))

ax.scatter(dt.index, dt);

plt.gcf().autofmt_xdate()