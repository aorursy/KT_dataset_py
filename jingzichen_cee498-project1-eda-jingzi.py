import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/cee-498-project1-london-bike-sharing/train.csv")

df
df.dropna(axis=0, how='any')

df.shape
df.dtypes
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['year'] = df['timestamp'].dt.year

df['month'] = df['timestamp'].dt.month

df['day'] = df['timestamp'].dt.day

df['hour'] = df['timestamp'].dt.hour

df.dtypes
quantitative = ['cnt', 't1', 't2', 'hum', 'wind_speed']

categorical = ['weather_code','is_holiday', 'is_weekend', 'season', 'year', 'month', 'day', 'hour']

df_quantitative = df[quantitative]

df_categorical = df[categorical]
df['cnt'].describe()
df['cnt'].hist()
df_quantitative.corr()
%config InlineBackend.figure_format = 'png'

sns.pairplot(df_quantitative)
fig, axes = plt.subplots(4, 2, sharey=True, figsize=(12, 32))

fig.subplots_adjust(hspace=0.5)

idx = 0

for x_temp in categorical:

    ax = axes[int(idx/2), idx%2]

    sns.barplot(x=x_temp, y='cnt', data=df, ax=ax)

    ax.set_xlabel(x_temp)

    ax.set_ylabel('cnt')

    idx = idx+1
fig, axes = plt.subplots(5, 1, sharey=True, figsize=(15, 32))

fig.subplots_adjust(hspace=0.5)

idx = 0

categorical_2 = ['weather_code','is_holiday', 'is_weekend', 'season', 'month']

for hue_temp in categorical_2:

    ax = axes[idx]

    sns.pointplot(x='hour', y='cnt', hue=hue_temp, data=df, ax=ax)

    ax.set_xlabel('hour')

    ax.set_ylabel('cnt')

    idx = idx+1
fig, axes = plt.subplots(3, 1, sharey=True, figsize=(15, 24))

fig.subplots_adjust(hspace=0.5)

idx = 0

categorical_2 = ['weather_code','is_holiday', 'is_weekend']

for hue_temp in categorical_2:

    ax = axes[idx]

    sns.pointplot(x='month', y='cnt', hue=hue_temp, data=df, ax=ax)

    ax.set_xlabel('month')

    ax.set_ylabel('cnt')

    idx = idx+1