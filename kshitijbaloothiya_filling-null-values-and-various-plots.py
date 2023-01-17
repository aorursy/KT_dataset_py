



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df=pd.read_csv("../input/housing-in-london/housing_in_london_monthly_variables.csv")

df.head()
df.isnull().sum()
df = df.dropna(subset = ['houses_sold'])

print(df.shape)

print(df.isnull().sum())
df['no_of_crimes'] = df['no_of_crimes'].fillna(df.groupby('area')['no_of_crimes'].transform('mean'))

london_crime = df['no_of_crimes'][0]

df = df.fillna(london_crime)

print(df.isnull().sum())

import matplotlib.pyplot as plt



y = df['average_price']

X_num = df[['no_of_crimes', 'houses_sold', 'borough_flag']]

df.plot(kind = 'line',  y = 'average_price')
time = df['date'].str.slice(0,4).astype(int)

time = pd.DataFrame(time)

df = df.assign(date=time)

import seaborn as sns

bp = df[['date', 'average_price']]

time = bp['date'].tolist()

price = bp['average_price'].tolist()

sns.boxplot(x =time ,y=price)
