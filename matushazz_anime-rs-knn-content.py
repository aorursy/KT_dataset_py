import pandas as pd

import numpy as np

import os
path = r'../input/'
anime = 'anime.csv'

ratings = 'rating.csv'
anime_path = os.path.join(path, anime)

ratings_path = os.path.join(path,ratings)
df_anime = pd.read_csv(anime_path)

df_ratings = pd.read_csv(ratings_path) 
df_anime.shape
df_ratings.shape
df_ratings.info()
df_anime.info()
n_users = df_ratings.user_id.unique().shape[0]

n_users
n_items = df_ratings.anime_id.unique().shape[0]

n_items
print(len(set(df_anime.anime_id)))
df_ratings.groupby('user_id').rating.count()
df_anime[df_anime['genre'].isnull()].sort_values('members', ascending = False).sample(10)
df_anime[df_anime['type'].isnull()].sort_values('members', ascending = False).sample(10)
df_anime[df_anime['rating'].isnull()].sort_values('members', ascending = False).sample(10)
df_anime = df_anime.replace('Unknown', np.nan)

df_anime = df_anime.dropna(how = 'all')

df_anime['type'] = df_anime['type'].fillna('TV')

df_anime['episodes'] = df_anime['episodes'].map(lambda x:np.nan if pd.isnull(x) else int(x))

df_ratings = df_ratings.replace(-1, np.nan)
df_anime[df_anime['anime_id']==841]
from matplotlib import pyplot as plt

import seaborn as sns; sns.set(style = 'white', palette = 'muted')
sns.pairplot(data=df_anime[['type','rating','episodes','members']].dropna(),hue='type')
%matplotlib inline

plt.hist(df_anime['rating'].fillna(0))
df_anime['rating'] = df_anime['rating'].fillna(df_anime.rating.median())
plt.hist(df_anime['rating'])
pd.DataFrame(df_ratings.groupby('rating').user_id.count()).reset_index()
sns.boxplot(data = df_anime, y = 'rating', x='type')
plt.hist(df_ratings.groupby(['anime_id'])['anime_id'].count())
sns.scatterplot( x = df_anime['episodes'], y= df_anime['rating'])
full_df = pd.merge(df_anime, df_ratings, how = 'right', on ='anime_id', suffixes = ['_avg', '_user'])

full_df.rename(columns = {'rating_user':'user_rating', 'rating_avg':'avg_rating'}, inplace = True)
full_df.sample(10)
df_col = full_df[['user_id', 'name', 'user_rating']]

df_col.head()
df_genres_list = df_anime['genre'].str.get_dummies(sep = ', ')
corr = df_genres_list.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df_genres_list.sample(10)
df_types_list = pd.get_dummies(df_anime[["type"]])

df_types_list.sample(10)
df_types_list.sample(10)
df_feat = df_anime[['members','rating','episodes']]
df_features = pd.concat([df_feat,df_genres_list, df_types_list], axis = 1).fillna(0)
df_anime[df_anime['anime_id']==5114]
def get_nombre_from_index(index):

    return df_anime[df_anime.index == index]['name'].values[0]

def get_id_from_nombre(name):

    return df_anime[df_anime.name == name]['anime_id'].values[0]

def get_index_from_id(anime_id):

    return df_anime[df_anime.anime_id == anime_id].index.values[0]
#Obtendremos el promedio de las valoraciones que el usuario ha dado a las series para determinar si le gustan, y le recomendaremos series similares a sus favoritas o mejor valoradas.. 

def get_user_top_list(user):

    df_user = df_ratings[df_ratings['user_id']==user]

    df_rated = df_user.dropna(how = 'any')

    avg =  df_rated.rating.mean() 

    df_toplist = df_rated[df_rated['rating']>= avg].sort_values('rating', ascending = False).head(10)

    return list(df_toplist['anime_id'])

def get_user_viewed_list(user):

    return list(df_ratings[df_ratings['user_id']==user]['anime_id'])
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import MaxAbsScaler
mas = MaxAbsScaler()

df_features2 = mas.fit_transform(df_features)
k = 11
neighbors_content = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree')
neighbors_content.fit(df_features2)
distances, indices = neighbors_content.kneighbors(df_features2)
distances.shape
indices.shape
from numpy import random
series = np,random.randint(0,len(indices))

print(series[1])

name = get_nombre_from_index(series[1])

print(name)
aid = get_id_from_nombre(name)
ind = get_index_from_id(aid)
anime = ind

list(indices[anime,1:11])
def get_recommendations(aid):

    anime =  get_index_from_id(aid)

    test = list(indices[anime,1:11])

    nb = []

    for i in test:

        a_name = get_nombre_from_index(i)

        nb.append(a_name)

    return nb
get_user_top_list(73509)
get_recommendations(23283)
def get_n_recommends(user, n):

    vistas = list(get_user_viewed_list(user))

    liked = list(get_user_top_list(user))

    lista = []

    for i in liked:

        ani = pd.Series(get_recommendations(i))

        recs = np.setdiff1d(ani, vistas) 

        lista.extend(recs)

        if(len(lista) > n):

            lista = lista[n:]

            break

    return lista
get_n_recommends(10,5)
get_n_recommends(73509, 10)