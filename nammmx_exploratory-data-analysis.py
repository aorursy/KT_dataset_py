import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent.csv')
df.head()
df.drop(columns=['Unnamed: 0'], inplace=True)
df['animal'] = df['animal'].map({

    'acept': 'yes',

    'not acept': 'no'

})
df['furniture'] = df['furniture'].map({

    'furnished': 'yes',

    'not furnished': 'no'

})
df[['hoa', 'rent amount', 'property tax', 'fire insurance', 'total']] = df[['hoa', 'rent amount', 'property tax', 'fire insurance', 'total']].apply(lambda x: x.str.lstrip('R$'))

df[['hoa', 'rent amount', 'property tax', 'fire insurance', 'total']] = df[['hoa', 'rent amount', 'property tax', 'fire insurance', 'total']].apply(lambda x: x.str.replace(',', ''))
df['hoa'] = pd.to_numeric(df['hoa'], errors='coerce')

df['rent amount'] = pd.to_numeric(df['rent amount'], errors='coerce')

df['property tax'] = pd.to_numeric(df['property tax'], errors='coerce')

df['fire insurance'] = pd.to_numeric(df['fire insurance'], errors='coerce')

df['total'] = pd.to_numeric(df['total'], errors='coerce')

df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
df.dtypes
df.isna().sum()
df['hoa'] = df['hoa'].fillna(df['hoa'].mean())

df['property tax'] = df['property tax'].fillna(df['property tax'].mean())

df['floor'] = df['floor'].fillna(df['floor'].mean()).astype(int)
df.head()
df.shape
plt.figure(figsize=(15,4))

sns.set(style="whitegrid")

sns.boxplot(x=df['total'])
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(12,6))

ax.scatter(df['total'], df['rent amount'])

ax.set_xlabel('total')

ax.set_ylabel('rent amount')

plt.show()
# skewness as measure for outliers

df['total'].skew()
# IQR method

q1 = df['total'].quantile(0.25)

q3 = df['total'].quantile(0.75)

IQR = q3 - q1

IF = q1 - (1.5 * IQR)

OF = q3 + (1.5 * IQR)
# show outliers

outlier = df[((df["total"] < IF) | (df["total"] > OF))]

len(outlier)
# remove outliers

df = df[~((df["total"] < IF) | (df["total"] > OF))]
df.shape
sns.set(style="whitegrid")

plt.figure(figsize=(10,5))

sns.boxplot(x=df['total'])
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(12,6))

ax.scatter(df['total'], df['rent amount'])

ax.set_xlabel('total')

ax.set_ylabel('rent amount')

plt.show()
df['total'].skew()
df.describe().round(2)
df.head()
df['city'].unique()
df['animal'].unique()
df['furniture'].unique()
sns.set(style="whitegrid")

sns.barplot(x = df['city'], y = df["total"])
sns.set(style="whitegrid")

sns.barplot(x = df['animal'], y = df["total"])
sns.set(style="whitegrid")

sns.barplot(x = df['furniture'], y = df["total"])
df['rooms'].unique()
df['bathroom'].unique()
df['parking spaces'].unique()
df['floor'].unique()
plt.figure(figsize=(16,8))

sns.boxplot(x=df["rooms"], y=df["total"])
plt.figure(figsize=(16,8))

sns.boxplot(x=df["bathroom"], y=df["total"])
plt.figure(figsize=(16,8))

sns.boxplot(x=df["parking spaces"], y=df["total"])
df['area2'] = pd.cut(df['area'], [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000])

df.groupby('area2').agg(['count', 'mean'])['total']
df.head()
plt.figure(figsize=(16,8))

sns.boxplot(x=df["area2"], y=df["total"])
df.dropna(how='any', inplace=True)
df.drop(columns=['area2'], inplace=True)
df.head()