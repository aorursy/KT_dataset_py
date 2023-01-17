import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
from fastai.collab import *

from fastai.tabular import *
user,item,title = 'userId','movieId','title'
path = untar_data(URLs.ML_SAMPLE)

path
os.listdir(path)
ratings = pd.read_csv(path/'ratings.csv')

ratings.head()
data = CollabDataBunch.from_df(ratings, seed=42)

data
y_range = [0,5.5]

learn = collab_learner(data, n_factors=50, y_range=y_range)

learn
learn.fit_one_cycle(3, 5e-3)
data.show_batch()
folder = '../input/ml-100k/'

path = Path(folder)

os.listdir(folder)
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,

                      names=[user,item,'rating','timestamp'])

print('ratings length: ', len(ratings))

ratings.head()
movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1', header=None,

                    names=[item, 'title', 'date', 'N', 'url', *[f'g{i}' for i in range(19)]])

movies.head()
rating_movie = ratings.merge(movies[[item, title]])

rating_movie.head()
data = CollabDataBunch.from_df(rating_movie, seed=42, valid_pct=0.1, item_name=title)

data
data.show_batch()
y_range = [0,5.5] # range of target variable

learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(5, 5e-3)
#learn.save('dotprod')
learn.model
g = rating_movie.groupby(title)['rating'].count()

top_movies = g.sort_values(ascending=False).index.values[:1000]

top_movies[:10]
movie_bias = learn.bias(top_movies, is_item=True)

movie_bias.shape
mean_ratings = rating_movie.groupby(title)['rating'].mean()

movie_ratings = [(b, i, mean_ratings.loc[i]) for i,b in zip(top_movies,movie_bias)]
item0 = lambda o:o[0]
sorted(movie_ratings, key=item0)[:15]
sorted(movie_ratings, key=lambda o: o[0], reverse=True)[:15]
movie_w = learn.weight(top_movies, is_item=True)

movie_w.shape
movie_pca = movie_w.pca(3)

movie_pca.shape
fac0,fac1,fac2 = movie_pca.t()

movie_comp = [(f, i) for f,i in zip(fac0, top_movies)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]
sorted(movie_comp, key=itemgetter(0))[:10]
movie_comp = [(f, i) for f,i in zip(fac1, top_movies)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]
sorted(movie_comp, key=itemgetter(0))[:10]
idxs = np.random.choice(len(top_movies), 50, replace=False)

idxs = list(range(50))

X = fac0[idxs]

Y = fac2[idxs]

plt.figure(figsize=(15,15))

plt.scatter(X, Y)

for i, x, y in zip(top_movies[idxs], X, Y):

    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)

plt.show()