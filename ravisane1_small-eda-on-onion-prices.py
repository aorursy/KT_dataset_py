import numpy as np

import pandas as pd

import seaborn as sn

import matplotlib.pyplot as plt
df = pd.read_csv('../input/market-price-of-onion-2020/Onion Prices 2020.csv')

df.head()
df.arrival_date = pd.to_datetime(df.arrival_date)

df.head()
print(df['market'].unique())
df_city = df.loc[df['market'] == 'Kurnool']
# Lets check if there are missing values. The data is from Government Of India so I hope data is fine.

df_city.isnull().sum()
fig = plt.figure(figsize=(20, 10))

plt.style.use('seaborn')

plt.scatter(x= df.loc[df['market']=='Kurnool'].arrival_date, y = df.loc[df['market']=='Kurnool'].min_price, c='green')

plt.scatter(x= df.loc[df['market']=='Kurnool'].arrival_date, y = df.loc[df['market']=='Kurnool'].max_price, c='red')

plt.scatter(x= df.loc[df['market']=='Kurnool'].arrival_date, y = df.loc[df['market']=='Kurnool'].modal_price, c='yellow')

plt.gcf().autofmt_xdate()

plt.show()
fig = plt.figure(figsize=(20, 10))

plt.style.use('seaborn')

plt.scatter(x= df.loc[df['market']=='Mangalore'].arrival_date, y = df.loc[df['market']=='Mangalore'].min_price, c='green')

plt.scatter(x= df.loc[df['market']=='Mangalore'].arrival_date, y = df.loc[df['market']=='Mangalore'].max_price, c='red')

plt.scatter(x= df.loc[df['market']=='Mangalore'].arrival_date, y = df.loc[df['market']=='Mangalore'].modal_price, c='yellow')

plt.gcf().autofmt_xdate()

plt.show()
plt.style.use('seaborn-bright')

fig = plt.figure(figsize=(20, 10))

plt.plot_date(x= df.loc[df['market']=='Kurnool'].arrival_date, y = df.loc[df['market']=='Kurnool'].min_price, c='green', label = 'Kurnool')

plt.plot_date(x = df.loc[df['market']=='Mangalore'].arrival_date, y = df.loc[df['market']=='Mangalore'].min_price, c='red', label = 'Mangalore')

plt.gcf().autofmt_xdate()

plt.legend(loc='upper left')

plt.show()