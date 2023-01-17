

import pandas as pd

import numpy as np

from surprise.model_selection import cross_validate

from surprise.model_selection import train_test_split

from surprise import accuracy

from surprise import Reader, Dataset, SVD 
reader = Reader()

ratings = pd.read_csv('../input/ratings_small.csv')

ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()
svd = SVD()

cross_validate(svd,data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
trainset, testset = train_test_split(data, test_size=.25)

svd.fit(trainset)
predictions = svd.test(testset)

accuracy.rmse(predictions)
ratings[ratings['userId'] == 1]
svd.predict(1, 302, 3)