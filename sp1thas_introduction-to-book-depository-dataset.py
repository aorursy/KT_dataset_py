import pandas as pd

import os

import json

from glob import glob

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
if os.path.exists('../input/book-depository-dataset'):

    path_prefix = '../input/book-depository-dataset/{}.csv'

else:

    path_prefix = '../export/kaggle/{}.csv'





df, df_f, df_a, df_c, df_p = [

    pd.read_csv(path_prefix.format(_)) for _ in ('dataset', 'formats', 'authors', 'categories', 'places')

]
# df = df.sample(n=500)

df.head()
df.describe()
df["publication-date"] = df["publication-date"].astype("datetime64")

df.groupby(df["publication-date"].dt.year).id.count().plot(title='Publication date distribution')
df["index-date"] = df["index-date"].astype("datetime64")

df.groupby(df["index-date"].dt.month).id.count().plot(title='Crawling date distribution')
df.groupby(['lang']).id.count().sort_values(ascending=False)[:5].plot(kind='pie', title="Most common languages")
import math

sns.lineplot(data=df.groupby(df['rating-avg'].dropna().apply(int)).id.count().reset_index(), x='rating-avg', y='id')
dims = pd.DataFrame({

    'dims': df['dimension-x'].fillna('0').astype(int).astype(str).str.cat(df['dimension-y'].fillna('0').astype(int).astype(str),sep=" x ").replace('0 x 0', 'Unknown').values, 

    'id': df['id'].values

})

dims.groupby(['dims']).id.count().sort_values(ascending=False)[:8].plot(kind='pie', title="Most common dimensions")
pd.merge(

    df[['id', 'publication-place']], df_p, left_on='publication-place', right_on='place_id'

).groupby(['place_name']).id.count().sort_values(ascending=False)[:8].plot(kind='pie', title="Most common publication places")