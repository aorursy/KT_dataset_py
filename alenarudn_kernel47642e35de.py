# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
datafr = pd.read_csv("../input/heart-disease-uci/heart.csv", error_bad_lines=False)
display(datafr.head(10))

from sklearn.model_selection import cross_val_score

def CrossVal(trainX,trainY,model):

    accuracy=cross_val_score(model,trainX , trainY, cv=10, scoring='accuracy')

    return(accuracy)

X= datafr.drop('target',axis=1)

Y=datafr['target']

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3,random_state=400)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=13, n_jobs=-1, random_state=40)

rf.fit(X_train,Y_train)

probs = rf.predict_proba(X_test)

probs = probs[:, 1]

predict5 = rf.predict(X_test)

score_ada= CrossVal(X_train,Y_train,rf)

print('Cross-Validation accuracy is {:.2f}%'.format(score_ada.mean()*100))
from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf.get_params())



{'bootstrap': True,

 'criterion': 'mse',

 'max_depth': None,

 'max_features': 'auto',

 'max_leaf_nodes': None,

 'min_impurity_decrease': 0.0,

 'min_impurity_split': None,

 'min_samples_leaf': 1,

 'min_samples_split': 2,

 'min_weight_fraction_leaf': 0.0,

 'n_estimators': 10,

 'n_jobs': 1,

 'oob_score': False,

 'random_state': 42,

 'verbose': 0,

 'warm_start': False}
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}

pprint(random_grid)
# Create a based model

# Instantiate the grid search model

from sklearn.model_selection import GridSearchCV



grid_search = GridSearchCV(estimator = rf, param_grid = random_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,Y_train)
print (grid_search.best_score_)