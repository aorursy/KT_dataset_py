import pandas as pd

import numpy as np
links = pd.read_csv('../input/links.csv')

movies = pd.read_csv('../input/movies.csv')

ratings = pd.read_csv('../input/ratings.csv')

tags = pd.read_csv('../input/tags.csv')
movies.head(3)
tags.head(3)
tags['tag'] = tags['tag'].apply(lambda x: x + '|')
tags = pd.DataFrame(tags.groupby('movieId')['tag'].sum())
tags.head(3)
movies_tags = movies.merge(tags, on='movieId', how='left')

movies_tags['tag'] = movies_tags['tag'].fillna('')
movies_tags.head(3)
movies_tags['description'] = movies_tags.apply(lambda x: x['genres'].replace('|', ' ') + ' ' + 

                                               x['tag'].replace('|', ' '), axis=1)

movies_tags = movies_tags.drop('genres', axis=1)

movies_tags = movies_tags.drop('tag', axis=1)
movies_tags.head(3)
movies_list = []

description_list = []



for mov, desc in movies_tags[['title', 'description']].values:

    movies_list.append(mov)

    description_list.append(desc.replace('r:', ' ').replace('.', ' ').replace('-', ' ').replace(':', ' '))
movies_tags.shape[0] == len(description_list)
description_list[:10]
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
coutn_v = CountVectorizer()
X_train = coutn_v.fit_transform(description_list)
X_train.toarray(), X_train.toarray().shape
tfidf = TfidfTransformer()

X_train_col = tfidf.fit_transform(X_train)
X_train_col.toarray(), X_train_col.toarray().shape
from tqdm import tqdm_notebook

for i in tqdm_notebook(range(X_train_col.shape[1])):

    col_name = 'd{}'.format(i)

    movies_tags[col_name] = pd.Series(X_train_col.toarray()[:, i])
movies_tags = movies_tags.drop('description', axis=1)
movies_tags.head(3)
train_data = movies_tags.iloc[:, 2:]
test_data = movies_tags[movies_tags['title'] == 'Jumanji (1995)'].iloc[:, 2:]
from sklearn.neighbors import NearestNeighbors
neighbor = NearestNeighbors(n_neighbors=5, n_jobs=-1, metric='euclidean')
neighbor.fit(train_data)
predict = neighbor.kneighbors(test_data, return_distance=True)
predict

movies.iloc[predict[1][0]]
ratings.head(3)
ratings.shape
mean_rating = ratings.groupby('userId')['rating'].mean()
ratings['mean_rating'] = ratings['userId'].apply(lambda x: mean_rating[x])
ratings['good_rating'] = ratings.apply(lambda x: x['rating'] if x['mean_rating'] <= 

                                       x['rating'] else np.NaN, axis=1)
ratings.head()
ratings = ratings[ pd.isnull( ratings['good_rating'] ) == 0 ]

ratings = ratings.drop('mean_rating', axis=1)

ratings = ratings.drop('good_rating', axis=1)

ratings = ratings.drop('timestamp', axis=1)
ratings.head(3)
ratings.shape
movies_ratings = ratings.merge(movies, on='movieId', how='left')

movies_ratings = movies_ratings.drop('movieId', axis=1)

movies_ratings = movies_ratings.drop('rating', axis=1)

movies_ratings = movies_ratings.drop('genres', axis=1)

movies_ratings.head(3) # оставили в таблице только то что нам пригодится
movies_ratings['userId'] = movies_ratings['userId'].apply(lambda x: str(x) + ' ')

movies_userid = movies_ratings.groupby('title')['userId'].sum()

movies_userid = movies_userid.reset_index(name='userId')
movies_userid.head()
movies_list = []

users_list = []



for mov, user in movies_userid[['title', 'userId']].values:

    movies_list.append(mov)

    users_list.append(user)
movies_userid.shape[0] == len(users_list)
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
coutn_v = CountVectorizer()
X_train = coutn_v.fit_transform(users_list)
X_train.toarray(), X_train.toarray().shape
tfidf = TfidfTransformer()

X_train_col = tfidf.fit_transform(X_train)
X_train_col.toarray(), X_train_col.toarray().shape
from tqdm import tqdm_notebook

for i in tqdm_notebook(range(X_train_col.shape[1])):

    col_name = 'd{}'.format(i)

    movies_userid[col_name] = pd.Series(X_train_col.toarray()[:, i])
movies_userid = movies_userid.drop('userId', axis=1)
movies_userid.head(3)
train_data = movies_userid.iloc[:, 1:]
test_data = movies_userid[movies_userid['title'] == 'Jumanji (1995)'].iloc[:, 1:]
neighbor.fit(train_data)
result = neighbor.kneighbors(test_data, return_distance=True)
result

movies.iloc[result[1][0]]