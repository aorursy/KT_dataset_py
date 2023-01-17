import numpy as np

import pandas as pd

import itertools

import collections

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/anime.csv',header=0)

cus_data = pd.read_csv('../input/rating.csv',header=0)
data.dtypes
cus_data.dtypes
data.describe()
cus_data.describe()
data = data.replace('Unknown',np.nan)

data = data.dropna()

data['episodes'] = data['episodes'].astype('int64')
sns.pairplot(data=data[['type','members','episodes','rating']],hue='type')
sns.boxplot(data=data,x='type',y='rating')
data['genre']=data['genre'].apply(lambda x : x.split(', '))

genre_data = itertools.chain(*data['genre'].values.tolist())

genre_counts = collections.Counter(genre_data)
genre_counts
genre = pd.DataFrame.from_dict(genre_counts,orient='index').reset_index().rename(columns={'index':'genre',0:'counts'})

genre = genre.sort_values('counts',ascending=False)
sns.barplot(y=genre['genre'],x=genre['counts'],color='skyblue')
def mapper(data,col):

    if col in data:

        return 1

    elif col not in data:

        return 0

genre_collections = pd.DataFrame([],columns=genre_counts.keys())

for col in genre_collections:

    genre_collections[col] = data['genre'].apply(mapper,args=(col,))
genre_collections.head()
sns.heatmap(genre_collections.corr())