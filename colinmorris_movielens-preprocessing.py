import random

from functools import lru_cache

import os

import math



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow import keras



import sklearn.preprocessing

from sklearn.preprocessing import LabelEncoder
# Hack for running on kernels and locally

RUNNING_ON_KERNELS = 'KAGGLE_WORKING_DIR' in os.environ

input_dir = '../input' if RUNNING_ON_KERNELS else '../input/movies'

out_dir = '.' if RUNNING_ON_KERNELS else '../input/movielens_preprocessed'



rating_path = os.path.join(input_dir, 'rating.csv')

df = pd.read_csv(rating_path, usecols=['userId', 'movieId', 'rating'])

# Shuffle (reproducibly)

df = df.sample(frac=1, random_state=1).reset_index(drop=True)



# Partitioning train/val according to behaviour of keras.Model.fit() when called with

# validation_split kwarg (which is to take validation data from the end as a contiguous

# chunk)

val_split = .05

n_ratings = len(df)

n_train = math.floor(n_ratings * (1-val_split))

itrain = df.index[:n_train]

ival = df.index[n_train:]



# Compactify movie ids. 

movie_id_encoder = LabelEncoder()

# XXX: Just fitting globally for simplicity. See movie_helpers.py for more 'principled'

# approach. I don't think there's any realistically useful data leakage here though.

#orig_movieIds = df['movieId']

df['movieId'] = movie_id_encoder.fit_transform(df['movieId'])



# Add centred target variable

df['y'] = df['rating'] - df.loc[itrain, 'rating'].mean()



SCALE = 0

if SCALE:

    # Add version of target variable scale to [0, 1]

    yscaler = sklearn.preprocessing.MinMaxScaler()

    yscaler.fit(df.loc[itrain, 'rating'].values.reshape(-1, 1))

    df['y_unit_scaled'] = yscaler.transform(df['rating'].values.reshape(-1, 1))



path = os.path.join(out_dir, 'rating.csv')

df.to_csv(path, index=False)
# Save a 10% sample of ratings for exercises (with re-compactified movieIds, and mapping back to canonical movie ids)

from sklearn.model_selection import GroupShuffleSplit



movie_counts = df.groupby('movieId').size()

thresh = 1000

pop_movies = movie_counts[movie_counts >= thresh].index



pop_df = df[df.movieId.isin(pop_movies)]



# Take approx 10% of the whole dataset

frac = 2 * 10**6 / len(pop_df)

print(frac)

splitter = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=1)

splits = splitter.split(pop_df, groups=pop_df.userId)

_, mini = next(splits)



mini_df = pop_df.iloc[mini].copy()



print(

    '{:,}'.format(len(mini_df)),

    len(df.userId.unique()) // 1000,

    len(mini_df.userId.unique()) // 1000,

    sep='\n',

)



# Compactify ids



def compactify_ids(df, col, backup=True):

    encoder = LabelEncoder()

    if backup:

        df[col+'_orig'] = df[col]

    df[col] = encoder.fit_transform(df[col])

    

for col in ['movieId', 'userId']:

    compactify_ids(mini_df, col, backup=col=='movieId')

    

# Shuffle

mini_df = mini_df.sample(frac=1, random_state=1)



# Recalculate y (just to be totally on the level. Very little opportunity for contamination here.)

val_split = .05

n_mini_train = math.floor(len(mini_df) * (1-val_split))

mini_train_rating_mean = mini_df.iloc[:n_mini_train]['rating'].mean()

mini_df['y'] = mini_df['rating'] - mini_train_rating_mean



path = os.path.join(out_dir, 'mini_rating.csv')

mini_df.to_csv(path, index=False)



print(

    df.userId.max(),

    mini_df.userId.max(),

    '\n',

    df.movieId.max(),

    mini_df.movieId.max(),

)
def munge_title(title):

    i = title.rfind(' (')

    if i != -1:

        title = title[:i]

    for suff_word in ['The', 'A', 'An']:

        suffix = ', {}'.format(suff_word)

        if title.endswith(suffix):

            title = suff_word + ' ' + title[:-len(suffix)]

    return title



def get_year(title):

    l = title.rfind('(') + 1

    try:

        return int(title[l:l+4])

    except ValueError:

        print(title, end='\t')

        return 0



movie_path = os.path.join(input_dir, 'movie.csv')

movie_df = pd.read_csv(movie_path)

mdf = movie_df



# XXX: hack

assert mdf.loc[

    mdf.movieId==64997,

    'title'].iloc[0] == 'War of the Worlds (2005)'

mdf.loc[

    mdf.movieId==64997,

    'title'

] = 'War of the Worlds (2005)x'



#mdf['movieId_orig'] = mdf['movieId']

n_orig = len(mdf)



# There are some movies listed in movie.csv which have no ratings. Drop them.

whitelist = set(movie_id_encoder.classes_)

mdf = mdf[mdf['movieId'].isin(whitelist)].copy()

print("Went from {} movies to {} after filtering out movies with no ratings".format(

    n_orig, len(mdf)

))



# New, compact movie Ids

mdf['movieId'] = movie_id_encoder.transform(mdf['movieId'].values)



mdf = mdf.sort_values(by='movieId').reset_index(drop=True)



# By default use original title field (which includes year of release) as unique key

mdf['key'] = mdf['title']



mdf['year'] = mdf['title'].map(get_year)

mdf['title'] = mdf['title'].map(munge_title)



# For movies whose munged title are unique, use it as their key

title_counts = mdf.groupby('title').size()

unique_titles = title_counts.index[title_counts == 1]

unique_ids = mdf.index[mdf.title.isin(unique_titles)]

mdf.loc[unique_ids, 'key'] = mdf.loc[unique_ids, 'title']



mdf['n_ratings'] = df.groupby('movieId').size()

mean_ratings = df.groupby('movieId')['rating'].mean()

mdf['mean_rating'] = mean_ratings



path = os.path.join(out_dir, 'movie.csv')

mdf.to_csv(path)
mdf.head()
df.head()