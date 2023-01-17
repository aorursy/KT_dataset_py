import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df_train = pd.read_csv('/kaggle/input/some-countries-ncov-data-credibility-tracking/covid_tracking_table.csv')

#df_train = df_train.replace(np.nan, '', regex=True) # replace nan with empty string

df_train.head(5)

df_train['date'] = pd.to_datetime(df_train['date'],infer_datetime_format=True) #convert from string to datetime

indexedDataset = df_train.set_index(['date'])

indexedDataset.head(5)
country = df_train.groupby('contury')['contury'].apply(set)

current_confirmed = df_train.groupby('contury').max().sort_values('confirm', ascending=False)

tops = current_confirmed[:10]

fig, ax = plt.subplots(figsize=(20, 10))

plt.xlabel('Date')

plt.ylabel('Number of confirmed cases')

for country, df in df_train.groupby('contury'):

    if country not in tops.index:

        continue

    #plt.plot(df['date'], df['confirm'], ls="-", lw=2, label="plot figure")

    c = df.query('confirm>0').set_index('date').sort_index()['confirm']

    c.plot(label=country, ax=ax)

    ax.annotate(country, xy=(c.index[-1], c.iloc[-1]), size=10)