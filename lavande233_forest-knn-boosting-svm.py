# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from operator import itemgetter

import numpy as np # linear algebra

import scipy

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

from scipy.stats import randint as sp_randint

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

from sklearn.svm import SVC

from time import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train
train = pd.read_csv("../input/train.csv")

train['color_id'] =train['color']

train['color_id'][train['color']=='white']=1

train['color_id'][train['color']=='clear']=2

train['color_id'][train['color']=='black']=3

train['color_id'][train['color']=='blue']=4

train['color_id'][train['color']=='blood']=5

train['color_id'][train['color']=='green']=6

train


from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

from operator import itemgetter

from scipy.stats import randint as sp_randint

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

import pickle



global param_dist



param_dist = {"max_depth": sp_randint(1,11),

              "min_samples_split": sp_randint(1, 11),

              "subsample": [0.1, 0.2, 0.5],

              "min_samples_leaf": sp_randint(1, 11),

              "learning_rate": [0.1, 0.01]}





# Utility function to report best scores

def report(grid_scores, n_top=10):

    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

    for i, score in enumerate(top_scores):

        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score,

                                                                     np.std(score.cv_validation_scores)),

              "Parameters: {0}".format(score.parameters))

    return





# Utility function to run Randomized_search

def RandomSearch(estimator, param_dist, X, y, n_iter_search=20):

    random_search = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=n_iter_search)

    start = time()

    random_search.fit(X, y)

    report(random_search.grid_scores_)

    print("")

    return random_search
param_dist = {'C': scipy.stats.expon(scale=100), 

              'gamma': scipy.stats.expon(scale=.1),

              'kernel': ['rbf'],

              'class_weight':['balanced', None]}

clf = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=20)

grid_scores = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type']).grid_scores_

report(grid_scores=grid_scores)
param_dist = {"n_neighbors":sp_randint(10,20),

             'weights':('uniform','distance')}



clf = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=param_dist, n_iter=20)

grid_scores = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type']).grid_scores_

report(grid_scores=grid_scores)

#Mean validation score: 0.704 (std: 0.016) Parameters: {'weights': 'distance', 'n_neighbors': 15}
param_dist = {"max_depth": sp_randint(1,11),

              "min_samples_split": sp_randint(2, 11),

              "subsample": [0.2, 0.5],

              "min_samples_leaf": sp_randint(5, 11),

              "learning_rate": [0.1, 0.01]}



clf = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=param_dist, n_iter=20)

grid_scores = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type']).grid_scores_

report(grid_scores=grid_scores)

# Mean validation score: 0.712 (std: 0.028) 

#Parameters: {'subsample': 0.2, 'learning_rate': 0.01, 'max_depth': 9, 'min_samples_leaf': 5, 'min_samples_split': 2}

# 0.709{'subsample': 0.5, 'learning_rate': 0.1, 'max_depth': 1, 'min_samples_leaf': 4, 'min_samples_split': 7}
param_dist = {'n_estimators':[10,20,50],

              'bootstrap':[True,False],

              'criterion':['gini','entropy'],

              "max_depth": sp_randint(1,11),

              "min_samples_split": sp_randint(2, 11),

              "min_samples_leaf": sp_randint(1, 11)}



clf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=40)

grid_scores = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type']).grid_scores_

report(grid_scores=grid_scores)
param_dist = {'n_estimators':[10,20,50,100],

              'bootstrap':[True,False],

              'criterion':['gini','entropy'],

              "max_depth": sp_randint(1,11),

              "min_samples_split": sp_randint(2, 11),

              "min_samples_leaf": sp_randint(1, 11)}



clf = RandomizedSearchCV(ExtraTreesClassifier(), param_distributions=param_dist, n_iter=40)

grid_scores = clf.fit(train[['bone_length','rotting_flesh','hair_length']],train['type']).grid_scores_

report(grid_scores=grid_scores)
obj = OneHotEncoder()

obj.fit(train['color_id'])

print(obj.transform(train['color_id']))