#!pip install scikit-surprise



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from surprise import Dataset, Reader

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



data = pd.read_csv("/kaggle/input/restaurant-data-with-consumer-ratings/rating_final.csv")
data.head()
data.shape
data.nunique()
data.rating.unique()
reader = Reader(line_format="user item rating", rating_scale = (0, 2))

#reader = Reader()

#restaurant_data = Dataset.load_from_df(data, reader)
restaurant_data = Dataset.load_from_df(data[['userID','placeID','rating']], reader)
print(restaurant_data)

print(type(restaurant_data))
restaurant_data.raw_ratings[0:10]
similarity_parameters = {

    'name' : 'cosine',

    'user_based': True,

    'min_support' : 3

}
from surprise import KNNWithMeans



KNN_Algo = KNNWithMeans(k=3, sim_options = similarity_parameters)
from surprise.model_selection import cross_validate



cross_validate(KNN_Algo, 

               restaurant_data, 

               measures=['RMSE', 'MAE'], 

               cv=5, 

               verbose=True)
# Use full data for training



trainset = restaurant_data.build_full_trainset()



KNN_Algo.fit(trainset)
# Getting data points where predictions can be made

testset = trainset.build_anti_testset()
# Making predictions

predictions = KNN_Algo.test(testset)
# Verify few predictions

predictions[0:4]
# Fetching top 10 predictions for each user

from collections import defaultdict



def get_top_n(predictions, n=10):

    '''Return the top-N recommendation for each user from a set of predictions.



    Args:

        predictions(list of Prediction objects): The list of predictions, as

            returned by the test method of an algorithm.

        n(int): The number of recommendation to output for each user. Default

            is 10.



    Returns:

    A dict where keys are user (raw) ids and values are lists of tuples:

        [(raw item id, rating estimation), ...] of size n.

    '''



    # First map the predictions to each user.

    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:

        top_n[uid].append((iid, est))



    # Then sort the predictions for each user and retrieve the k highest ones.

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key=lambda x: x[1], reverse=True)

        top_n[uid] = user_ratings[:n]



    return top_n



from itertools import islice



def take(n, iterable):

    "Return first n items of the iterable as a list"

    return list(islice(iterable, n))



top_n = get_top_n(predictions, n=10)

take(10, top_n.items())
# Printing top predictions

for uid, user_ratings in take(10,top_n.items()):

    print(uid, [iid for (iid, _) in user_ratings])
places = pd.read_csv("/kaggle/input/restaurant-data-with-consumer-ratings/geoplaces2.csv")
places = places.set_index('placeID')

places.head()
# Printing top predictions

for uid, user_ratings in take(5,top_n.items()):

    print("For User",uid)

    for  (iid, _) in user_ratings:

        print(iid)

        ids = iid-1

        print(places.loc[iid,"name"])
from surprise.model_selection import GridSearchCV



sim_options = {

    "name": ["msd", "cosine"],

    "min_support": [3, 4, 5],

    "user_based": [False, True],

}



param_grid = {"sim_options": sim_options}



jokes_gs = GridSearchCV(KNNWithMeans, 

                  param_grid, 

                  measures=["rmse", "mae"], 

                        cv=3)



jokes_gs.fit(restaurant_data)



print(jokes_gs.best_score["rmse"])

print(jokes_gs.best_params["rmse"])
from surprise import SVD

from surprise import Dataset,accuracy

from surprise.model_selection import cross_validate

from surprise.model_selection import train_test_split



# Load the in-built movielens-100k dataset (download it if needed).

#ml_data = Dataset.load_builtin('ml-100k')



# sample random trainset and testset

# test set is made of 25% of the ratings.

#trainset, testset = train_test_split(ml_data, test_size=.25)



# We'll use the famous SVD algorithm.

SVD_Algo = SVD()



# Train the algorithm on the trainset, and predict ratings for the testset

SVD_Algo.fit(trainset)

predictions = SVD_Algo.test(testset)



# Then compute RMSE

accuracy.rmse(predictions)
%%time

import random                                                              

                                                                           

# Load your full dataset.                                                  

#ml_data = Dataset.load_builtin('ml-100k')                                     

raw_ratings = restaurant_data.raw_ratings                                             

                                                                           

# shuffle ratings if you want                                              

random.shuffle(raw_ratings)                                                

                                                                           

# 90% trainset, 10% testset                                                

threshold = int(.9 * len(raw_ratings))                                     

trainset_raw_ratings = raw_ratings[:threshold]                             

test_raw_ratings = raw_ratings[threshold:]                                 

                                                                           

restaurant_data.raw_ratings = trainset_raw_ratings  # data is now your trainset                                                           

                                                                           

# Select your best algo with grid search. Verbosity is buggy, I'll fix it. 

print('GRID SEARCH BEGIN...')                                                    

param_grid = {

    "n_epochs": [5, 10],

    "lr_all": [0.002, 0.005],

    "reg_all": [0.4, 0.6]

}



movie_gs = GridSearchCV(SVD, 

                        param_grid, 

                        measures=["rmse", "mae"], 

                        cv=3)



movie_gs.fit(restaurant_data)

print('GRID SEARCH END...')                                                    
ml_final = movie_gs.best_estimator['rmse']                                  

                                                                           

# retrain on the whole train set                                           

trainset = restaurant_data.build_full_trainset()                                      

ml_final.fit(trainset)                                                       

                                                                           

# now test on the trainset                                                 

testset = restaurant_data.construct_testset(trainset_raw_ratings)                     

predictions = ml_final.test(testset)                                           

print('Accuracy on the trainset:')                                         

accuracy.rmse(predictions)                                                 

                                                                           

# now test on the testset                                                  

testset = restaurant_data.construct_testset(test_raw_ratings)                         

predictions = ml_final.test(testset)                                           

print('Accuracy on the testset:')                                          

accuracy.rmse(predictions)
predictions[0:10]
top_n = get_top_n(predictions, n=10)

take(10, top_n.items())
# Printing top predictions

for uid, user_ratings in take(5,top_n.items()):

    print("For User",uid)

    for  (iid, _) in user_ratings:

        print(iid)

        ids = iid-1

        print(places.loc[iid,"name"])