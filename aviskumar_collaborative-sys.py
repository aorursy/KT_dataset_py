from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import pandas as pd

import numpy as np
import os

os.listdir('../input')
ratings = pd.read_csv("../input/collaborative-system/ratings_sub.csv",encoding = "ISO-8859-1")

ratings.head()
ratings.shape
ratings.userId=ratings.userId.astype(str)

ratings.movieId=ratings.movieId.astype(str)
ratings.columns
# Total unique users 

print("total unique users - ",len(ratings["userId"].unique()))
# Users with max no of movies watches

ratings["userId"].value_counts().head()
from surprise import Dataset,Reader
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'title', 'rating']], reader)
data
# Split data to train and test

from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=.25,random_state=123)



# to build on full data

#trainset = data.build_full_trainset()
type(trainset)
# user item rating data can be obtained as follows

user_records = trainset.ur

type(user_records)
for keys in user_records.keys():

    print(keys)
user_records[0]
# However the ids are the inner ids and not the raw ids

# raw ids can be obatined as follows



print(trainset.to_raw_uid(0))

print(trainset.to_raw_iid(1066))
user_records[0]
from surprise import KNNWithMeans

from surprise import accuracy
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson', 'user_based': False})

algo.fit(trainset)



#Item Item similarity matrix has been created now
len(testset)
testset[0:5]
# Evalute on test set

test_pred = algo.test(testset)



# compute RMSE

accuracy.rmse(test_pred)
# View a particular prediction

test_pred[12]



# To access a particular value, say estimate simply mention test_pred[12].est



#was_impossible= true means the data si sparse ie. it dont had neccessry info in matrix
test_pred[12].details["actual_k"]
# convert results to dataframe

test_pred_df = pd.DataFrame(test_pred)

test_pred_df["was_impossible"] = [x["was_impossible"] for x in test_pred_df["details"]]
test_pred_df.loc[test_pred_df.was_impossible].head(5)
# Mkae prediction for a single user

algo.predict(uid="user_405",iid="Wrong Trousers, The (1993)")



#Here the recommendated rating is None, because was_impossible is true
#was_impossible=false: are only calculated



testset_new = trainset.build_anti_testset()
len(testset_new)
testset_new[0:5]
predictions = algo.test(testset_new[0:10000])
predictions_df = pd.DataFrame([[x.uid,x.est] for x in predictions])
predictions_df.columns = ["userId","est_rating"]

predictions_df.sort_values(by = ["userId", "est_rating"],ascending=False,inplace=True)
predictions_df.head()
top_10_recos = predictions_df.groupby("userId").head(10).reset_index(drop=True)
# Lets exclude movies with very few ratings, say less than 5

movie_count = ratings["title"].value_counts(ascending=False)

pop_movie = movie_count.loc[movie_count.values > 200].index

len(pop_movie)

ratings = ratings.loc[ratings.title.isin(pop_movie)]

ratings.shape
from surprise import Dataset,Reader

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(ratings[['userId', 'title', 'rating']], reader)
ratings.shape
# Split data to train and test

from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=.25,random_state=123)



# to build on full data

#trainset = data.build_full_trainset()
from surprise import SVD

from surprise import accuracy
svd_model = SVD(n_factors=50,biased=False)

svd_model.fit(trainset)
test_pred = svd_model.test(testset)


# compute RMSE

accuracy.rmse(test_pred)
user_factors = svd_model.pu

user_factors.shape

item_factors = svd_model.qi

item_factors.shape
pred = np.dot(user_factors,np.transpose(item_factors))
pred[1523,0:5]
svd_model.predict(uid = trainset.to_raw_uid(1523), iid = trainset.to_raw_iid(0))
from surprise.model_selection import GridSearchCV

param_grid = {'n_factors' : [5,10,15], "reg_all":[0.01,0.02]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3,refit = True)

gs.fit(data)
# get all parameter combinations

gs.param_combinations
# get best parameters

gs.best_params
# Use the "best model" for prediction

gs.test(testset)
import numpy as np
item_factors
item_sim = np.corrcoef(item_factors)

max_val = (-item_sim).argsort()
topk = pd.DataFrame(max_val[:,0:20])
# create item iid dictionary



all_movies = [trainset.to_raw_iid(x) for x in range(0,567)]

movie_iid_dict = dict(zip(range(0,567), all_movies))
topk = topk.replace(movie_iid_dict)
topk["movie"] = all_movies
topk.to_csv("sim_movies_svd.csv",index=False)