# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from scipy import spatial



a = [1, 2]

b = [2, 4]

c = [2.5, 4]

d = [4.5, 5]



print('c and a', spatial.distance.euclidean(c, a))



print('c and b', spatial.distance.euclidean(c, b))



print('c and d', spatial.distance.euclidean(c, d))
a = [1, 2]

b = [2, 4]

c = [2.5, 4]

d = [4.5, 5]



print('c and a', spatial.distance.cosine(c,a))

print('c and b', spatial.distance.cosine(c,b))

print('c and d', spatial.distance.cosine(c,d))

print('a ans b', spatial.distance.cosine(a,b))

# load_data.py



from surprise import Dataset

from surprise import Reader



# This is the same data that was plotted for similarity earlier

# with one new user "E" who has rated only movie 1

ratings_dict = {

    "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],

    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],

    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],

}



df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))



# Loads Pandas dataframe

data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

# Loads the builtin Movielens-100k data

movielens = Dataset.load_builtin('ml-100k')
movielens
# recommender.py



from surprise import KNNWithMeans



# To use item-based cosine similarity

sim_options = {

    "name": "cosine",

    "user_based": False,  # Compute  similarities between items

}

algo = KNNWithMeans(sim_options=sim_options)
trainingSet = data.build_full_trainset()
algo.fit(trainingSet)



prediction = algo.predict('E', 2)

prediction.est
from surprise.model_selection import GridSearchCV



data = Dataset.load_builtin("ml-100k")

sim_options = {

    "name": ["msd", "cosine"],

    "min_support": [3, 4, 5],

    "user_based": [False, True],

}



param_grid = {"sim_options": sim_options}



gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)

gs.fit(data)



print(gs.best_score["rmse"])

print(gs.best_params["rmse"])
from surprise import SVD



data = Dataset.load_builtin("ml-100k")



param_grid = {

    "n_epochs": [5, 10],

    "lr_all": [0.002, 0.005],

    "reg_all": [0.4, 0.6]

}

gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)



gs.fit(data)



print(gs.best_score["rmse"])

print(gs.best_params["rmse"])
from surprise.model_selection import cross_validate



# Load the movielens-100k dataset (download it if needed).

#data = Dataset.load_builtin('ml-100k')



# Use the famous SVD algorithm.

algo = SVD()



# Run 5-fold cross-validation and print results.

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
from surprise import accuracy

from surprise.model_selection import train_test_split



# sample random trainset and testset

# test set is made of 25% of the ratings.

trainset, testset = train_test_split(data, test_size=.25)



# We'll use the famous SVD algorithm.

algo = SVD()



# Train the algorithm on the trainset, and predict ratings for the testset

algo.fit(trainset)

predictions = algo.test(testset)



# Then compute RMSE

accuracy.rmse(predictions)
from surprise import NormalPredictor

from surprise import Reader



# Creation of the dataframe. Column names are irrelevant.

ratings_dict = {'itemID': [1, 1, 1, 2, 2],

                'userID': [9, 32, 2, 45, 'user_foo'],

                'rating': [3, 2, 4, 3, 1]}

df = pd.DataFrame(ratings_dict)



# A reader is still needed but only the rating_scale param is requiered.

reader = Reader(rating_scale=(1, 5))



# The columns must correspond to user id, item id and ratings (in that order).

data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)



# We can now use this dataset as we please, e.g. calling cross_validate

cross_validate(NormalPredictor(), data, cv=2)
df