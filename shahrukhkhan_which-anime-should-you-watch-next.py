import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
## loading datasets

anime_df = pd.read_csv('../input/anime-recommendations-database/anime.csv')

anime_df.head()
rating_df = pd.read_csv('../input/anime-recommendations-database/rating.csv')
rating_df.head()
anime_df.describe()
times_rated = rating_df.groupby(['anime_id'])['rating'].count()

times_rated = times_rated.rename('times_rated')
times_rated.describe()
rating_df = rating_df.merge(times_rated,on='anime_id')
## selecting top 50% rated animes

rating_df_top50 = rating_df[rating_df['times_rated']>51]
rating_df_top50[['rating','times_rated']].describe()
anime_rating_df = anime_df.merge(rating_df_top50,on='anime_id')
anime_rating_df.head()
anime_pivot_df = pd.pivot_table(index='name',columns='user_id',values='rating_y', data=anime_rating_df)



anime_pivot_df.fillna(value=0,inplace=True)
anime_pivot_df.head()
from scipy.sparse import csr_matrix

anime_mat = csr_matrix(anime_pivot_df.values)
from sklearn.neighbors import NearestNeighbors

anime_nbrs = NearestNeighbors(metric='cosine', algorithm='auto').fit(anime_mat)

distances, indices = anime_nbrs.kneighbors(anime_mat)
## creating list for anime names

anime_names = list(anime_pivot_df.index)
kimi_no_nawa_index = anime_names.index('Kimi no Na wa.')

distances, indices = anime_nbrs.kneighbors(anime_pivot_df.iloc[kimi_no_nawa_index,:].values.reshape(1,-1),n_neighbors=6)
indices_flat, distances_flat = indices.flatten(),distances.flatten()

for index,anime_index in enumerate(indices_flat):

    anime_name = anime_names[anime_index]

    if(index == 0): 

        print(f'Animes similar to {anime_name}:')

    else:

        print(f'\t {anime_name} with score ---> {distances_flat[index]}')