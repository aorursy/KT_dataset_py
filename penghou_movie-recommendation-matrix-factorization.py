import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from surprise import Reader, Dataset, SVD, evaluate, accuracy
ratings = pd.read_csv('../input/ratings.csv')
ratings.head()
print(f'number of movies: {ratings["movieId"].nunique()}')
print(f'number of users: {ratings["userId"].nunique()}')
print(f'number of ratings: {len(ratings)}')
links = pd.read_csv('../input/links.csv')
metadata = pd.read_csv('../input/movies_metadata.csv')
reader = Reader()
svd = SVD()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd.fit(trainset)
predict_ratings = (np.dot(svd.pu[0:1], svd.qi.T) + svd.bu[0] + svd.bi + trainset.global_mean)[0]
item_innerid_to_rawid_map = dict((v,k) for k,v in trainset._raw2inner_id_items.items())
pd.Series(predict_ratings, index=list(item_innerid_to_rawid_map.values())).sort_values(ascending=False).head(10)
np.dot(svd.pu, svd.qi.T) + svd.bu + svd.bi + trainset.global_mean