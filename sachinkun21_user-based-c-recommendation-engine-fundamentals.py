# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
ratings  = pd.read_csv('../input/ratings_small.csv')

ratings.head()
print(ratings.shape)
from sklearn.model_selection import train_test_split

train_df,test_df = train_test_split(ratings, test_size = 0.3, random_state = 42)

print(train_df.shape, '\t\t', test_df.shape)
train_df.head()
df_movies_as_features = train_df.pivot(index = 'userId', columns = 'movieId',values = 'rating' )

df_movies_as_features.shape
df_movies_as_features.head()
df_movies_as_features.fillna(0, inplace = True)

df_movies_as_features.head()
dummy_train=train_df.copy()

dummy_test =test_df.copy()
dummy_train.rating.value_counts()
dummy_test.rating.value_counts()
dummy_train['rating'] =dummy_train.rating.apply(lambda x : 0  if x >=1 else 1)

dummy_test['rating'] =dummy_test.rating.apply(lambda x : 1  if x >=1 else 0)

dummy_train.head()
def pivot_by_movie(df):

    df = df.pivot(index='userId', columns = 'movieId', values = 'rating')

    df.fillna(0, inplace = True)

    return df
dummy_train = pivot_by_movie(dummy_train)

dummy_test = pivot_by_movie(dummy_test)

dummy_train.head()
from sklearn.metrics.pairwise import pairwise_distances
user_correlation  = 1- pairwise_distances(df_movies_as_features,metric = 'cosine')

user_correlation.shape
user_correlation
np.sum(np.isnan(user_correlation))
user_correlation[np.isnan(user_correlation)]=0

user_correlation
train_movies_as_feature = train_df.pivot(index='userId', columns = 'movieId', values = 'rating')

train_movies_as_feature.head()
train_movies_as_feature.shape
mean = np.nanmean(train_movies_as_feature, axis = 1)

print(mean.shape)
normalised_df = (train_movies_as_feature.T-mean).T

normalised_df.head()
user_correlation =  1 - pairwise_distances(normalised_df.fillna(0), metric ='cosine')

user_correlation[np.isnan(user_correlation)]=0

print(user_correlation)

user_correlation.shape
user_correlation[user_correlation<0]=0
user_predicted_ratings = np.dot(user_correlation, train_movies_as_feature.fillna(0))

user_predicted_ratings
user_predicted_ratings.shape
user_final_rating = np.multiply(user_predicted_ratings, dummy_train)

user_final_rating.head()
user_final_rating.iloc[670].sort_values(ascending =False)[-6:]