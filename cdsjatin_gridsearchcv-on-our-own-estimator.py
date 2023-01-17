# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.base import BaseEstimator

import numpy as np

from sklearn.metrics import accuracy_score



class random_class(BaseEstimator):

    def __init__(self, const=0, use_const=0):

        self.y = []

        self.cons = const

        self.use_cons = use_const

        

    def fit(self, X, y):

        self.y = y

        return self

    

    def score(self, X, y):

        np.asarray(X)

        n = np.shape(X)[0]

        

        ops = np.random.choice(self.y, n, replace=True) if self.use_cons else [self.cons]*n

        return np.random.choice([0.1, 0.2,0.5])
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_wine
data = load_wine()



X, y = data['data'], data['target']
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()



params = {

    'max_depth': [4, 8],

    'max_features': [2, 4]

}



gsv = GridSearchCV(dt, params, cv=5).fit(X, y)



gsv.best_params_



gsv.verbose = 1



gsv.cv_results_
params = {'const': [1, 2], 

          'use_const': [0, 1]}



rdc = random_class()



gsv = GridSearchCV(rdc, params, cv=5).fit(X, y)
gsv.cv_results_
gsv.best_params_
gsv.best_score_