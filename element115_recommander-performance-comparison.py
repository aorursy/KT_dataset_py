import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import random



# Note that there are no NANs in these data; '?' is

# used when there is missing information

rating = pd.read_csv('../input/rating_final.csv')
rating.head()
# Surprise's Input is a df of this shape :

overall_rating = rating[['userID','placeID','rating']]

overall_rating.head()
from surprise import SVD,SVDpp,KNNBasic,KNNWithZScore

from surprise.dataset import Reader, Dataset

from surprise.model_selection import LeaveOneOut

from surprise import accuracy
reader = Reader(rating_scale=(0, 2))

ds = Dataset.load_from_df(overall_rating,reader)

loo = LeaveOneOut(n_splits=1,min_n_ratings=1)
LR = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
train_loss = []

test_loss = []

models = []

for i in range(len(LR)) :

    lr_all = LR[i]

    algo = SVD(n_epochs=50,reg_all=0.01,lr_all=lr_all)

    models.append(algo)

    for trainset,testset in loo.split(ds) : #train - validation split with leave one out

        # train and test algorithm.

        algo.fit(trainset)

        train_pred = algo.test(trainset.build_testset())

        test_pred = algo.test(testset)



        # Compute and print Root Mean Squared Error

        train_rmse = accuracy.rmse(train_pred, verbose=False)

        test_rmse = accuracy.rmse(test_pred, verbose=False)

        train_loss.append(train_rmse)

        test_loss.append(test_rmse)
plt.plot(LR,train_loss,label='train')

plt.plot(LR,test_loss, label = 'test')

plt.xlabel('learning_rate')

plt.ylabel('rmse')

plt.legend()
# Index of minimum element

i = test_loss.index(min(test_loss))

# using the best model

algo = models[i]

# predicting rating

algo.predict('U1077','132825')
from surprise.model_selection import GridSearchCV

param_grid = {

    'n_factors' : [10, 20, 50, 100, 130, 150, 200],

    'n_epochs': [10, 15, 30, 50, 100], 

    'lr_all': [0.001, 0.005, 0.007, 0.01, 0.05, 0.07, 0.1],

    'reg_all': [0.01, 0.05, 0.07, 0.1, 0.2, 0.4, 0.6]

}
gs_svd = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=loo,return_train_measures = True)

gs_svdpp = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=loo,return_train_measures = True)
gs_svd.fit(ds)

gs_svdpp.fit(ds)
# best RMSE score

print ("Best RMSE Scores")

print(f'SVD : {gs_svd.best_score["rmse"]}')

print(f'SVDpp : {gs_svdpp.best_score["rmse"]}')





# combination of parameters that gave the best RMSE score

print("Parameters")

print(f"SVD : {gs_svd.best_params['rmse']}")

print(f"SVDpp : {gs_svdpp.best_params['rmse']}")
results_frame_svd = pd.DataFrame.from_dict(gs_svd.cv_results)

results_frame_svd['model'] = 'SVD'

results_frame_svdpp = pd.DataFrame.from_dict(gs_svdpp.cv_results)

results_frame_svdpp['model'] = 'SVDpp'

results_frame = pd.concat([results_frame_svd, results_frame_svdpp])
results_frame.sort_values(by='mean_test_rmse').head()
sim_options = {'name': ['pearson', 'cosine', 'msd'],

               'user_based':[ True, False] # compute  similarities between users

               }

param_grid_knn = {

    'sim_options' : sim_options,

    'k' : [10, 20, 40, 100],

    'min_k' : [1, 5 , 10],

    'verbose' : [False]

}
gs_knn= GridSearchCV(KNNBasic, param_grid=param_grid_knn, measures=['rmse', 'mae'], cv=loo,return_train_measures = True)

gs_knnZ= GridSearchCV(KNNWithZScore, param_grid=param_grid_knn, measures=['rmse', 'mae'], cv=loo,return_train_measures = True)
gs_knn.fit(ds)

gs_knnZ.fit(ds)
# best RMSE score

print ("Best RMSE Scores")

print(f'KNN : {gs_knn.best_score["rmse"]}')

print(f'KNNWithZScore : {gs_knnZ.best_score["rmse"]}')





# combination of parameters that gave the best RMSE score

print("Parameters")

print(f"KNN : {gs_knn.best_params['rmse']}")

print(f"KNNWithZScore : {gs_knnZ.best_params['rmse']}")
results_frame_knn = pd.DataFrame.from_dict(gs_knn.cv_results)

results_frame_knn['model'] = 'KNN'

results_frame_knnZ = pd.DataFrame.from_dict(gs_knnZ.cv_results)

results_frame_knnZ['model'] = 'KNNWithZScore'

results_frame = pd.concat([results_frame, results_frame_knn, results_frame_knnZ],sort=False)
results_frame['rank_test_mae'] = results_frame['mean_test_mae'].rank()

results_frame['rank_test_rmse'] = results_frame['mean_test_rmse'].rank()

results_frame.head()
# Export for visualization

results_frame.to_csv('results.csv',index=False)