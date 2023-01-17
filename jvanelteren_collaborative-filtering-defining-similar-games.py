import pandas as pd

import pickle

import numpy as np

from fastai.collab import *

from pprint import pprint

import matplotlib.pyplot as plt

import umap

from scipy import stats

from sklearn.neighbors import NearestNeighbors

%matplotlib inline
# the original csv from https://raw.githubusercontent.com/beefsack/bgg-ranking-historicals/master/

# The column ID is used in API calls to retrieve the game reviews

games = pd.read_csv('../input/2019-05-02.csv')

games.describe()

games.sort_values('Users rated',ascending=False,inplace=True)

games.rename(index=str, columns={"Bayes average": "Geekscore",'Name':'name'}, inplace=True)
# load the file I composed with all the reviews

reviews = pd.read_csv('../input/bgg-13m-reviews.csv',index_col=0)

print(len(reviews))

reviews.head()
games_by_all_users = reviews.groupby('name')['rating'].agg(['mean','count']).sort_values('mean',ascending=False)

games_by_all_users['rank']=games_by_all_users.reset_index().index+1

print(len(games_by_all_users))
data = CollabDataBunch.from_df(reviews, user_name='user',item_name='name',rating_name='rating',bs=100000, seed = 42)

data.show_batch()
learner = collab_learner(data, n_factors=50, y_range=(2.,10))
lr_find(learner)

learner.recorder.plot()
learner.fit_one_cycle(3, 1e-2, wd=0.15)
#learner.save('5cycles7e-2-bs100000factors50yrange0-105')

#learner.save('4cycles3e-2-bs100000factors50yrange0-105wd03')

#learner.save('4cycles2e-2-bs100000factors20yrange1-10wd01')

#learner.save('3cycles1e-2-bs100000factors50yrange2-10wd005')

#learner.load('3cycles1e-2-bs100000factors50yrange2-10wd005')

learner.model
learner.recorder.plot_losses()
mean_ratings = reviews.groupby('name')['rating'].mean().round(2)

top_games = games_by_all_users[games_by_all_users['count']>5000].sort_values('mean',ascending=False).index

print(len(top_games))

game_bias = learner.bias(top_games, is_item=True)

game_bias.shapemean_ratings = reviews.groupby('name')['rating'].mean()

game_ratings = [(b, i, mean_ratings.loc[i]) for i,b in zip(top_games,game_bias)]

item0 = lambda o:o[0]
sorted(game_ratings, key=item0)[:10]
sorted(game_ratings, key=lambda o: o[0], reverse=True)[:15]
game_weights = learner.weight(top_games, is_item=True)

game_weights.shape
game_pca = game_weights.pca(3)

game_pca.shape
fac0,fac1,fac2 = game_pca.t()

game_comp = [(f, i) for f,i in zip(fac0, top_games)]

print('highest on this dimension')

pprint(sorted(game_comp, key=itemgetter(0), reverse=True)[:10]) 

print('lowest on this dimension')

pprint(sorted(game_comp, key=itemgetter(0), reverse=False)[:10]) 
game_comp = [(f, i) for f,i in zip(fac1, top_games)]

print('highest on this dimension')

pprint(sorted(game_comp, key=itemgetter(0), reverse=True)[:10])

print('lowest on this dimension')

pprint(sorted(game_comp, key=itemgetter(0), reverse=False)[:10])
game_comp = [(f, i) for f,i in zip(fac2, top_games)]

print('highest on this dimension')

pprint(sorted(game_comp, key=itemgetter(0), reverse=True)[:10])

print('lowest on this dimension')

pprint(sorted(game_comp, key=itemgetter(0), reverse=False)[:10])
idxs = np.random.choice(len(top_games), 50, replace=False)

idxs = list(range(50))

X = fac0[idxs]

Y = fac1[idxs]

plt.figure(figsize=(15,15))

plt.scatter(X, Y)

for i, x, y in zip(top_games[idxs], X, Y):

    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)

plt.show()
CUTOFF=2000

top_games = games_by_all_users[games_by_all_users['count']>CUTOFF].sort_values('mean',ascending=False).reset_index()

number_of_games = len(top_games)

print(number_of_games)

game_weights = learner.weight(top_games['name'], is_item=True)

game_bias = learner.bias(top_games['name'], is_item=True)

npweights = game_weights.numpy()

top_games['model_score']=game_bias.numpy()

top_games['weights_sum']=np.sum(np.abs(npweights),axis=1)



nn = NearestNeighbors(n_neighbors=number_of_games)

fitnn = nn.fit(npweights)
res = top_games[top_games['name']=='Chess']

if len(res)==1:

    distances,indices = fitnn.kneighbors([npweights[res.index[0]]])

else:

    print(res.head())

top_games.iloc[indices[0][:10]].sort_values('model_score',ascending=False)
res = top_games[top_games['name']=='Catan']

if len(res)==1:

    distances,indices = fitnn.kneighbors([npweights[res.index[0]]])

else:

    print(res.head())

top_games.iloc[indices[0][:10]].sort_values('model_score',ascending=False)
res = top_games[top_games['name']=='Agricola']

if len(res)==1:

    distances,indices = fitnn.kneighbors([npweights[res.index[0]]])

else:

    print(res.head())

top_games.iloc[indices[0][:10]].sort_values('model_score',ascending=False)