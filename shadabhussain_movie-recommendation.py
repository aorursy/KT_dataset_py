path = '/kaggle/input/movielens-small-dataset/ml-latest-small/ml-latest-small/'
# import libraties

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Reading ratings file

ratings = pd.read_csv(path + 'ratings.csv', encoding='latin-1')
ratings.head()
from sklearn.model_selection import train_test_split

train, test = train_test_split(ratings, test_size=0.30, random_state=31)
print(train.shape)

print(test.shape)
# pivot ratings into movie features

df_movie_features = train.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).fillna(0)
df_movie_features.head()
dummy_train = train.copy()

dummy_test = test.copy()
dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x>=1 else 1)

dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x>=1 else 0)
# The movies not rated by user is marked as 1 for prediction. 

dummy_train = dummy_train.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).fillna(1)



# The movies not rated by user is marked as 0 for evaluation. 

dummy_test = dummy_test.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).fillna(0)
dummy_train.head()
dummy_test.head()
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

user_correlation = 1 - pairwise_distances(df_movie_features, metric='cosine')

user_correlation[np.isnan(user_correlation)] = 0

print(user_correlation)
user_correlation.shape
movie_features = train.pivot(

    index='userId',

    columns='movieId',

    values='rating'

)
movie_features.head()
mean = np.nanmean(movie_features, axis=1)

df_subtracted = (movie_features.T-mean).T
df_subtracted.head()
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')

user_correlation[np.isnan(user_correlation)] = 0

print(user_correlation)
user_correlation[user_correlation<0]=0

user_correlation
user_predicted_ratings = np.dot(user_correlation, movie_features.fillna(0))

user_predicted_ratings
user_predicted_ratings.shape
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

user_final_rating.head()
user_final_rating.iloc[1].sort_values(ascending=False)[0:5]
movie_features = train.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).T



movie_features.head()
mean = np.nanmean(movie_features, axis=1)

df_subtracted = (movie_features.T-mean).T
df_subtracted.head()
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')

item_correlation[np.isnan(item_correlation)] = 0

print(item_correlation)
item_correlation[item_correlation<0]=0

item_correlation
item_predicted_ratings = np.dot((movie_features.fillna(0).T),item_correlation)

item_predicted_ratings
item_predicted_ratings.shape
dummy_train.shape
item_final_rating = np.multiply(item_predicted_ratings,dummy_train)

item_final_rating.head()
item_final_rating.iloc[1].sort_values(ascending=False)[0:5]
test_movie_features = test.pivot(

    index='userId',

    columns='movieId',

    values='rating'

)

mean = np.nanmean(test_movie_features, axis=1)

test_df_subtracted = (test_movie_features.T-mean).T



# User Similarity Matrix

test_user_correlation = 1 - pairwise_distances(test_df_subtracted.fillna(0), metric='cosine')

test_user_correlation[np.isnan(test_user_correlation)] = 0

print(test_user_correlation)
test_user_correlation[test_user_correlation<0]=0

test_user_predicted_ratings = np.dot(test_user_correlation, test_movie_features.fillna(0))

test_user_predicted_ratings
test_user_final_rating = np.multiply(test_user_predicted_ratings,dummy_test)
test_user_final_rating.head()
from sklearn.preprocessing import MinMaxScaler

from numpy import *



X  = test_user_final_rating.copy() 

X = X[X>0]



scaler = MinMaxScaler(feature_range=(1, 5))

print(scaler.fit(X))

y = (scaler.transform(X))



print(y)
test_ = test.pivot(

    index='userId',

    columns='movieId',

    values='rating'

)
# Finding total non-NaN value

total_non_nan = np.count_nonzero(~np.isnan(y))
rmse = (sum(sum((test_ - y )**2))/total_non_nan)**0.5

print(rmse)
test_movie_features = test.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).T



mean = np.nanmean(test_movie_features, axis=1)

test_df_subtracted = (test_movie_features.T-mean).T



test_item_correlation = 1 - pairwise_distances(test_df_subtracted.fillna(0), metric='cosine')

test_item_correlation[np.isnan(test_item_correlation)] = 0

test_item_correlation[test_item_correlation<0]=0
test_item_correlation.shape
test_movie_features.shape
test_item_predicted_ratings = (np.dot(test_item_correlation, test_movie_features.fillna(0))).T

test_item_final_rating = np.multiply(test_item_predicted_ratings,dummy_test)

test_item_final_rating.head()
test_ = test.pivot(

    index='userId',

    columns='movieId',

    values='rating'

)
from sklearn.preprocessing import MinMaxScaler

from numpy import *



X  = test_item_final_rating.copy() 

X = X[X>0]



scaler = MinMaxScaler(feature_range=(1, 5))

print(scaler.fit(X))

y = (scaler.transform(X))





test_ = test.pivot(

    index='userId',

    columns='movieId',

    values='rating'

)



# Finding total non-NaN value

total_non_nan = np.count_nonzero(~np.isnan(y))
rmse = (sum(sum((test_ - y )**2))/total_non_nan)**0.5

print(rmse)