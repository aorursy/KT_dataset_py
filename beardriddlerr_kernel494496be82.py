# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ratings_df = pd.read_csv('../input/u.data', sep='\t', 

                         names=['user_id', 'movie_id', 'rating', 'ts'])

ratings_df
ratings_df['rating'].max()
movie_df = pd.read_csv('../input/u.item', sep='|', encoding = "latin-1", header=None)

movie_df = movie_df[[0, 1]]

movie_df.columns = ['movie_id', 'movie_name']

movie_df
user_df = pd.read_csv('../input/u.user', sep='|', encoding = "latin-1", header=None)

user_df = user_df[[0, 1]]

user_df.columns = ['user_id', 'age']

user_df
mean_ratings = ratings_df.groupby('movie_id').agg({'rating': 'mean'}).reset_index().rename(

    {'rating': 'mean_rating'}, axis=1)

count_ratings = ratings_df.groupby('movie_id').agg({'rating': 'count'}).reset_index().rename(

    {'rating': 'count_rating'}, axis=1)

mean_ratings
base_model_df = movie_df.merge(mean_ratings).merge(count_ratings)

base_model_df
import matplotlib.pyplot as plt

import seaborn as sns
sns.scatterplot(x='count_rating', y='mean_rating', data=base_model_df)
sns.lmplot(x='count_rating', y='mean_rating', data=base_model_df)
base_model_df.loc[base_model_df['mean_rating'].argmax()]
min_count_base = base_model_df[base_model_df['count_rating'] > 50]

min_count_base.sort_values('mean_rating', ascending=False)
user_movie_ratings = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')

movie_user_ratings = ratings_df.pivot_table(index='movie_id', columns='user_id', values='rating')

user_movie_ratings
user_movie_ratings[1]
movie_corr = user_movie_ratings.corrwith(ratings_pivot_zeros[1]).reset_index()

movie_corr[movie_corr['movie_id']!=1][0].argmax()
movie_df[movie_df['movie_id']==340]
from scipy.spatial.distance import cosine
user_movie_rating = user_movie_ratings.fillna(0).values

movie_user_rating = user_movie_rating.T
user_movie_rating
movie_user_rating
from tqdm import tqdm



# movie_similarity = []

# for movie1 in tqdm(movie_user_rating):

#     movie_similarity1 = []

#     for movie2 in tqdm(movie_user_rating):

#         movie_similarity1.append(1 - cosine(movie1, movie2))

#     movie_similarity.append(movie_similarity1)

# movie_similarity = np.array(movie_similarity)

    

from sklearn.metrics.pairwise import cosine_similarity

movie_similarity = cosine_similarity(movie_user_rating)
movie_id = 1

user_id = 5

def item_colab_filtering(user_id, movie_id):

    return (user_movie_rating[user_id - 1] * movie_similarity[movie_id - 1]).sum() / movie_similarity[movie_id - 1].sum()
def get_item_user(user_id):

    P_ui = []

    for movie_id in movie_df['movie_id']:

        P_ui.append(item_colab_filtering(user_id, movie_id))

    return np.array(P_ui).argsort()[::-1]
get_item_user(10)
# set index, movie_name

movie_df_search = movie_df.set_index('movie_id')

movie_df_search.loc[710]
def get_name(movie_ids):

    return [movie_df_search.loc[id+1] for id in movie_ids]
get_name(get_item_user(10))