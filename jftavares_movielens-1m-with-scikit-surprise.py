from __future__ import (absolute_import, division, print_function,

                        unicode_literals)



import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from surprise import Dataset

from surprise import Reader

from surprise.model_selection import cross_validate

from surprise.model_selection import GridSearchCV

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore

from surprise import SVD, SVDpp, NMF

from surprise import SlopeOne, CoClustering

from surprise import accuracy

from surprise.model_selection import train_test_split

import os

reviews = pd.read_csv('../input/ml-1m/ml-1m/ratings.dat', names=['userID', 'movieID', 'rating', 'time'], delimiter='::', engine= 'python')

print('Rows:', reviews.shape[0], '; Columns:', reviews.shape[1], '\n')



reviews.head()
print('No. of Unique Users    :', reviews.userID.nunique())

print('No. of Unique Movies :', reviews.movieID.nunique())

print('No. of Unique Ratings  :', reviews.rating.nunique())
rts_gp = reviews.groupby(by=['rating']).agg({'userID': 'count'}).reset_index()

rts_gp.columns = ['Rating', 'Count']
plt.barh(rts_gp.Rating, rts_gp.Count, color='royalblue')

plt.title('Overall Count of Ratings', fontsize=15)

plt.xlabel('Count', fontsize=15)

plt.ylabel('Rating', fontsize=15)

plt.grid(ls='dotted')

plt.show()
file_path = os.path.expanduser('../input/ml-1m/ml-1m/ratings.dat')

reader = Reader(line_format='user item rating timestamp', sep='::')

data = Dataset.load_from_file(file_path, reader=reader)



trainset, testset = train_test_split(data, test_size=.15)
algoritmo = KNNBasic(k=50, sim_options={'name': 'pearson', 'user_based': True, 'verbose' : True})
algoritmo.fit(trainset)
uid = str(49)  
iid = str(2058)  # raw item id
print("Prediction for rating: ")

pred = algoritmo.predict(uid, iid, r_ui=4, verbose=True)
test_pred = algoritmo.test(testset)
print("Deviation RMSE: ")

accuracy.rmse(test_pred, verbose=True)
# Avalia MAE

print("Analisys MAE: ")

accuracy.mae(test_pred, verbose=True)
# KNNWithMeans with 50 neighbors, user based

algoritmo = KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': False, 'verbose' : True})



algoritmo.fit(trainset)



# Hide the real rating and try to predict

# real rating is 4

# Select User and Movie

uid = str(49)

iid = str(2058)



# Predict the rating

print("\nMaking prediction")

pred = algoritmo.predict(uid, iid, r_ui=4, verbose=True)



test_pred = algoritmo.test(testset)



# Deviation RMSE

print("\nDeviation RMSE: ")

accuracy.rmse(test_pred, verbose=True)



# Analisys MAE

print("\nAnalisys MAE: ")

accuracy.mae(test_pred, verbose=True)