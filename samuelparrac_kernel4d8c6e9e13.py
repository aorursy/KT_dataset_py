# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import os

import itertools

import collections

import operator

import collections

from matplotlib import pyplot as plt

import seaborn as sns; sns.set(style = 'white', palette = 'muted')

from numpy import random

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import MaxAbsScaler
import pandas as pd

df_animes = pd.read_csv("../input/anime-recommendations-database/anime.csv")

df_ratings = pd.read_csv("../input/anime-recommendations-database/rating.csv")
df_animes.shape
df_animes.info()
df_animes.head()
print(df_animes.isnull().sum())

print(df_animes[df_animes.isnull().any(axis=1)].shape)
df_animes.head()
df_animes[df_animes['rating'].isnull()].sort_values('members', ascending = False).sample(10)
df_animes[df_animes['genre'].isnull()].sort_values('members', ascending = False).sample(10)
anime_list = df_animes.anime_id.unique()

rated_anime_list = df_ratings.anime_id.unique()

print('cantidad de animes: {}, animes evaluados: {}'.format(len(anime_list), len(rated_anime_list)))
df_animes = df_animes.replace('Unknown', np.nan)

df_animes = df_animes.dropna(how = 'all')

df_animes['episodes'] = df_animes['episodes'].map(lambda x:np.nan if pd.isnull(x) else int(x))

#df_ratings = df_ratings.replace(-1, np.nan)

df_animes['type'] = df_animes['type'].fillna('TV')
df_animes.info()
sns.pairplot(data=df_animes[['type','rating','episodes','members']].dropna(),hue='type')
sns.boxplot(data = df_animes, y = 'rating', x='type')
df_animes.groupby('type').anime_id.size()
plt.figure(figsize = (16,5))

df_check=df_animes[['type','rating','episodes','members']].corr()

sns.heatmap(df_check, annot=True, fmt="g", cmap='viridis')

plt.show()
plt.hist(df_animes['rating'].fillna(0))
df_animes['rating'] = df_animes['rating'].fillna(df_animes.rating.median())
plt.hist(df_animes['rating'].fillna(0))
# List de generos

genres = set()

for entry in df_animes['genre']:

    if not type(entry) is str:

        continue

    genres.update(entry.split(", "))

print(genres)

print("Total Genres: " + str(len(genres)))
# List genres by count

genres_count = collections.defaultdict(int)

for entry in df_animes['genre']:

    if not type(entry) is str:

        continue

    seen_already = set()

    for genre in entry.split(", "):

        if genre in seen_already:

            continue

        seen_already.add(genre)

        genres_count[genre] += 1

sorted(genres_count.items(), key=operator.itemgetter(1), reverse=True)
fig = plt.figure(figsize=(20,20))

ax = plt.gca()

plt.title('Analisis a nivel de Genero por rating y cantidad')

plt.xlabel('Rating')

plt.ylabel('Cantidad')

num_colors = len(genres)

cm = plt.get_cmap('gist_rainbow')

ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])

ax.set_yscale('log')



for genre in genres:

    data_genre = df_animes[df_animes.genre.str.contains(genre) == True]

    ax.plot(data_genre["rating"], data_genre["members"], marker='o', linestyle='', ms=12, label=genre)

ax.legend(numpoints=1, loc='upper left');
df_ratings.shape
df_ratings.info()
user_list = df_ratings.user_id.unique()

print('usuarios unicos: {}'.format(len(user_list)))
df_ratings['rating'].value_counts().plot(kind='bar', legend='Reverse')
df_ratings[df_ratings['rating'] == -1].groupby('anime_id')['anime_id'].count().sort_values(ascending = False).head()
df_animes[df_animes['anime_id'] == 1535]
plt.hist(df_ratings[df_ratings['rating'] != -1].groupby('anime_id')['anime_id'].count())
df_ratings = df_ratings.replace(-1, np.nan)
df_ratings['rating'].value_counts().plot(kind='bar', legend='Reverse')
df_genero_list = df_animes['genre'].str.get_dummies(sep = ', ')
df_genero_list.sample(5)
df_types_list = pd.get_dummies(df_animes[["type"]])

df_types_list.sample(5)
df_feat_num = df_animes[['members','rating','episodes']]
df_feat_num.info()
df_features_target = pd.concat([df_feat_num ,df_genero_list, df_types_list], axis = 1).fillna(0)
df_features_target.head()
df_features_target.info()
def get_nombre_from_index(index):

    return df_animes[df_animes.index == index]['name'].values[0]

def get_id_from_nombre(name):

    return df_animes[df_animes.name == name]['anime_id'].values[0]

def get_index_from_id(anime_id):

    return df_animes[df_animes.anime_id == anime_id].index.values[0]
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import MaxAbsScaler
mas = MaxAbsScaler()

df_features_target1 = mas.fit_transform(df_features_target)
k = 7
neighbors_content = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree')
neighbors_content.fit(df_features_target1)
distances, indices = neighbors_content.kneighbors(df_features_target)
distances.shape
def get_user_top_list(user):

    df_user = df_ratings[df_ratings['user_id']==user]

    df_rated = df_user.dropna(how = 'any')

    avg =  df_rated.rating.mean() 

    df_toplist = df_rated[df_rated['rating']>= avg].sort_values('rating', ascending = False).head(10)

    return list(df_toplist['anime_id'])

def get_user_viewed_list(user):

    return list(df_ratings[df_ratings['user_id']==user]['anime_id'])
#Generamos una lista unica de series

def get_unique_series(series):

    newlist=list(set(series))

    return (newlist)
#Selecciona el grupo de animes más cercanos al consultado

def get_recommendations(aid):

    anime =  get_index_from_id(aid)

    test = list(indices[anime,1:11])

    nb = []

    for i in test:

        a_name = get_nombre_from_index(i)

        nb.append(a_name)

    return nb
def get_n_recommends(user, n):

    vistas = list(get_user_viewed_list(user))

    liked = list(get_user_top_list(user))

    series = []

    for i in liked:

        ani = pd.Series(get_recommendations(i))

        recs = np.setdiff1d(ani, vistas) 

        series.extend(recs)

        newlist=get_unique_series(series)

        if(len(newlist) > n):

            series = newlist[:n]

            break

    return newlist
list(get_user_viewed_list(7816))
list(get_user_top_list(7816))
get_recommendations(2966)
user=list(df_ratings["user_id"].sample(1))
user
get_n_recommends(25380,5)
list(get_user_top_list(7816))