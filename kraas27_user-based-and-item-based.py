import pandas as pd

import numpy as np

from surprise import KNNBasic, KNNWithMeans, Dataset, Reader, accuracy

from surprise.model_selection import train_test_split, GridSearchCV
movies = pd.read_csv('../input/movies.csv')

ratings = pd.read_csv('../input/ratings.csv')
movies.head(3)
ratings.head(3)
movies_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)

movies_ratings.dropna(inplace=True)

movies_ratings.head()
dataset = pd.DataFrame({

    'uid': movies_ratings.userId,

    'iid': movies_ratings.title,

    'rating': movies_ratings.rating

})
dataset.head() 
ratings.rating.min()
ratings.rating.max()
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(dataset, reader)
params = {'k':np.arange(10, 101, 10),

          'sim_options': {'name': ['pearson_baseline'], 'user_based': [True]}

         }

grid_algo = GridSearchCV(KNNBasic, params, measures=['rmse', 'mae'], cv=5, n_jobs=-1)

grid_algo.fit(data)
grid_algo.best_params
trainset, testset = train_test_split(data, test_size=.15)
algo = KNNBasic(k=40, sim_options={'name': 'pearson_baseline', 'user_based': True})

algo.fit(trainset)
predict = algo.test(testset)
accuracy.rmse(predict, verbose=True)
algo = KNNWithMeans(k=30, sim_options={'name': 'pearson_baseline', 'user_based': False})

algo.fit(trainset)
test_pred = algo.test(testset)
accuracy.rmse(test_pred, verbose=True)