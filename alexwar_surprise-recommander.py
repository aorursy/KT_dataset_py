import time
import datetime
import random

import numpy as np
import six
from tabulate import tabulate

from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
import numpy as np
import pandas as pd
from scipy import sparse
from time import time
from numpy import matrix
from numpy.random import rand
import re
%%time
def load_and_clean (): 
    anime =  pd.read_csv("../input/anime.csv", sep=",")
    rating = pd.read_csv("../input/rating.csv", sep=",")
    
    anime["episodes"] = anime["episodes"].map(lambda x:np.nan if x=="Unknown" else x)
    anime["type"] = anime["type"].map(lambda x:np.nan if x=="Unknown" else x)
    anime["genre"] = anime["genre"].map(lambda x:np.nan if x=="Unknown" else x)
    anime["episodes"].fillna(anime["episodes"].median(),inplace = True)
    anime["name"] = anime["name"].map(lambda name:re.sub('[^A-Za-z0-9]+', " ", name))
    anime["rating"] = anime["rating"].astype(float) 
    anime.rating.replace({-1: np.nan}, regex=True, inplace = True)
    rating.rating.replace({-1: np.nan}, regex=True, inplace = True)
    rating.dropna(inplace = True)
    
    return anime, rating

anime, rating  = load_and_clean()


def merge_it_all(df1, df2, n_sample) :
    df = pd.merge(df1, df2, left_on ="anime_id", right_on = "anime_id")
    df = df[df["user_id"] <= n_sample]
    return df

df = merge_it_all(anime, rating, 5000)

def get_unbiased_rating(df) : 
    users_interactions_count_df = df.groupby(['user_id', 'anime_id']).size().groupby('user_id').size()
    print('# users: %d' % len(users_interactions_count_df))
    users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 8].reset_index()[['user_id']]
    print('# users with at least 8 interactions: %d' % len(users_with_enough_interactions_df))
    print('# of interactions: %d' % len(df))
    interactions_from_selected_users_df = df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'user_id',
               right_on = 'user_id')
    print('# of interactions from users with at least 8 interactions: %d' % len(interactions_from_selected_users_df))
    popularity = interactions_from_selected_users_df.groupby('anime_id').size().reset_index(name='popularity')
    interactions_full_df = pd.merge(popularity, interactions_from_selected_users_df)
    interactions_full_df.rename({"rating_y" : "user_rating"}, inplace = True, axis = 1)
    return interactions_full_df, interactions_from_selected_users_df

interactions_full_df, interactions_from_selected_users_df = get_unbiased_rating(df)
interactions_full_df.head(5)
from surprise import Reader 
reader = Reader()
data = Dataset.load_from_df(interactions_full_df[['user_id', 'anime_id', 'user_rating']], reader)
data.split(n_folds=3)
from surprise import NormalPredictor
normal_pred =  NormalPredictor()
from surprise import BaselineOnly
B0 =  BaselineOnly()
from surprise import KNNBasic
KNNbasic =  KNNBasic()
from surprise import KNNWithMeans
mean_knn = KNNWithMeans()
from surprise import KNNBaseline
KNNbaseline = KNNBaseline()
from surprise import SVD
svd = SVD()
from surprise import SVDpp
svdpp = SVDpp()
from surprise import NMF
NMF_model =  NMF()
from surprise import SlopeOne
Slop =  SlopeOne()
from surprise import CoClustering
CClus =  CoClustering()

from surprise import evaluate

for reco in [svdpp, Slop, CClus, KNNbaseline, mean_knn, 
             KNNbasic, NMF_model, normal_pred, B0] :
    evaluate(reco, data, measures=['RMSE', 'MAE'])
