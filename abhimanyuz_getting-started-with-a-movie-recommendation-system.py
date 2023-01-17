import pandas as pd 

import numpy as np 
from surprise import Reader, Dataset, SVD, evaluate

reader = Reader()

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

df_movies=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
df_movies.columns
df_movies.shape
df_movies.head()
ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

data.split(n_folds=5)
svd = SVD()

evaluate(svd, data, measures=['RMSE'])
trainset = data.build_full_trainset()

svd.fit(trainset)
ratings[ratings['userId'] == 1]
df_movies.loc[df_movies['id'] > '300']
svd.predict(1, 1339, 3).est