%matplotlib inline

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

# import surprise
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import evaluate
from surprise import CoClustering
from surprise import KNNBasic
from surprise.prediction_algorithms.matrix_factorization import SVD, SVDpp, NMF
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.similarities import cosine, msd, pearson, pearson_baseline
from surprise.dataset import Trainset

# jester_items = pd.read_table('jester_dataset_2/jester_items.dat')
jester_ratings = pd.read_table('../input/jester_ratings.dat')
jester_ratings.head()
# jester_items = pd.read_csv('jester_dataset_2/jester_items.dat', sep='p', header=None)
jester_ratings = pd.read_csv('../input/jester_ratings.dat', sep='\s+', header=None)
# jester_items.head()
jester_ratings.columns = ["User ID", "Item ID", "Rating"]
jester_ratings.head()
# jester_items.columns = ["joke no.", "joke"]
# jester_items.head()
jj = jester_ratings.loc[:, "User ID"]
jj.describe()
jj.unique().size
reader = Reader(line_format='user item rating timestamp', sep='\t',rating_scale = (-10, 10))
jester = Dataset.load_from_df(jester_ratings, reader=reader)
print(jester)
# j = jester_ratings.dropna(any)
some_values = {8,104} #range (8,105)
j2 = jester_ratings.loc[jester_ratings['Item ID'].isin(some_values)]

some_values_1 = {8} #range (8,105)
some_values_2 = {104} #range (8,105)
j2_1 = jester_ratings.loc[jester_ratings['Item ID'].isin(some_values_1)]
j2_2 = jester_ratings.loc[jester_ratings['Item ID'].isin(some_values_2)]
#drop items 8 to 104
# j2_1.describe()
j2.sort_values(by=['Item ID'])
j2.head()
j2_1.sort_values(by=['Item ID'])
j2_1.head()
j2_2.sort_values(by=['Item ID'])
j2_2.head()
j3 = j2
j3.head()
jester_test = j2.loc[:, "User ID" : "Item ID"]
jester_test.head()
reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale = (-10, 10))
jester_actual_test = Dataset.load_from_df(j3, reader=reader)
trainset, testset = train_test_split(jester, test_size=.25)
fullset, noset = train_test_split(jester, test_size=0.0001)
train = jester.build_full_trainset()
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }

algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

predictions = algo.test(testset)

print(accuracy.rmse(predictions))

#presented a good RMSE value
# We'll use CoClustering Basic algorithm.
algo = CoClustering()
algo.fit(trainset)

predictions = algo.test(testset)

accuracy.rmse(predictions)
algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)

print(accuracy.rmse(predictions))
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }

algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

predictions = algo.test(testset)

print(accuracy.rmse(predictions))

#cross_validate(algo, jester, cv=2)
df1 = j2_1.set_index('User ID')
df2 = j2_2.set_index('User ID')
df1.head()
df2.head()
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }

algo = KNNBasic(k =200, k_min = 300, sim_options=sim_options)

algo.fit(train)

# cross_validate(algo, jester, cv=2)

# predictions = algo.test(jester_actual_test)

# print(accuracy.rmse(predictions))

for i in range(1, 59133):
    uid = str(i)  # raw user id (as in the ratings file). They are **strings**!
#     other = pd.DataFrame({'User ID': uid})

    if((j2_1['User ID'] == i).any()):
        iid = str(8)  # raw item id (as in the ratings file). They are **strings**!
        # get a prediction for specific users and items.
        pred = algo.predict(train.to_inner_uid(i), train.to_inner_iid(8), r_ui=df1.loc[i,'Rating'], verbose=True)
    
    if((j2_2['User ID'] == i).any()):
        iid = str(104)  # raw item id (as in the ratings file). They are **strings**!
        # get a prediction for specific users and items.
        pred = algo.predict(train.to_inner_uid(i), train.to_inner_iid(104), r_ui=df2.loc[i,'Rating'], verbose=True)
    
sim_options = {'name': 'msd',
               'user_based': False   # no shrinkage
               }
algo = KNNBasic(k =200, k_min = 300, sim_options=sim_options)

algo.fit(train)

# cross_validate(algo, jester, cv=2)

# predictions = algo.test(jester_actual_test)

# print(accuracy.rmse(predictions))

for i in range(1, 59133):
    uid = str(i)  # raw user id (as in the ratings file). They are **strings**!
#     other = pd.DataFrame({'User ID': uid})

    if((j2_1['User ID'] == i).any()):
        iid = str(8)  # raw item id (as in the ratings file). They are **strings**!
        # get a prediction for specific users and items.
        pred = algo.predict(uid, iid, r_ui=df1.loc[i,'Rating'], verbose=True)
    
    if((j2_2['User ID'] == i).any()):
        iid = str(104)  # raw item id (as in the ratings file). They are **strings**!
        # get a prediction for specific users and items.
        pred = algo.predict(uid, iid, r_ui=df2.loc[i, 'Rating'], verbose=True)
    

sim_options = {'name': 'pearson',
               'user_based': True  # make user based
               }
algo = KNNBasic(k = 200, k_min = 300, sim_options=sim_options)

algo.fit(train)

# cross_validate(algo, jester, cv=2)

# predictions = algo.test(jester_actual_test)

# print(accuracy.rmse(predictions))

for i in range(1, 59133):
    uid = str(i)  # raw user id (as in the ratings file). They are **strings**!
#     other = pd.DataFrame({'User ID': uid})

    if((j2_1['User ID'] == i).any()):
        iid = str(8)  # raw item id (as in the ratings file). They are **strings**!
        # get a prediction for specific users and items.
        pred = algo.predict(uid, iid, r_ui=df1.loc[i,'Rating'], verbose=True)
    
    if((j2_2['User ID'] == i).any()):
        iid = str(104)  # raw item id (as in the ratings file). They are **strings**!
        # get a prediction for specific users and items.
        pred = algo.predict(uid, iid, r_ui=df2.loc[i, 'Rating'], verbose=True)