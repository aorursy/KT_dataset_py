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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score

from mlxtend.classifier import StackingCVClassifier

from sklearn.model_selection import cross_validate
import pandas as pd

train = pd.read_csv('../input/learn-together/train.csv', index_col = 'Id')

test = pd.read_csv('../input/learn-together/test.csv', index_col = 'Id')

train.columns

train.info()

train.head()
# Make a copy of train df for ML experiments

train_2 = train.copy()

train_2.columns
# Separate feature and target arrays as X and y

X = train_2.drop('Cover_Type', axis = 1)

y=train_2.Cover_Type

print(X.columns)

y[:5]
# Parameters for Random Forest Hyperparameter tuning

param_grid_rf = {'n_estimators': np.logspace(2,3.5,8).astype(int),

                 'max_features': [0.1,0.3,0.5,0.7,0.9],

                 'max_depth': np.logspace(0,3,10).astype(int),

                 'min_samples_split': [2, 5, 10],

                 'min_samples_leaf': [1, 2, 4],

                 'bootstrap':[True, False]}



# Instantiate RandomForestClassifier

rf = RandomForestClassifier(random_state = 42)



# Create a Random Search Parameter Grid

grid_rf = RandomizedSearchCV(estimator=rf, 

                          param_distributions=param_grid_rf, 

                          n_iter=100,

                          cv=3, 

                          verbose=3, 

                          n_jobs=1,

                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 

                          refit='NLL')

# Fit Random Forest Models - 100 models each with 3 rounds of cross validation - 300 fitted models

grid_rf.fit(X,y)

print('The Best Random Forest Estimator is: ', grid_rf.best_estimator_)

print('The Best Random Forest Parameters are: ', grid_rf.best_params_)

print('The Best Random Forest score is: ', grid_rf.best_score_)
# Parameters for Extremely Randomized Trees Hyperparameter tuning

param_grid_extra = {'n_estimators': np.logspace(2,3.5,8).astype(int),

                    'max_features': [0.1,0.3,0.5,0.7,0.9],

                    'max_depth': np.logspace(0,3,10).astype(int),

                    'min_samples_split': [2, 5, 10],

                    'min_samples_leaf': [1, 2, 4],

                    'bootstrap':[True, False]}



# Instantiate ExtraTreesClassifier

extra_trees = ExtraTreesClassifier(random_state = 42)



# Create a Random Search Parameter Grid

grid_extra_trees = RandomizedSearchCV(estimator=extra_trees, 

                          param_distributions=param_grid_extra, 

                          n_iter=100,

                          cv=3, 

                          verbose=3, 

                          n_jobs=1,

                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 

                          refit='NLL')

# Fit Extra Trees Models - 100 models each with 3 rounds of cross validation - 300 fitted models

grid_extra_trees.fit(X,y)

print('The Best Extra Trees Estimator is: ', grid_extra_trees.best_estimator_)

print('The Best Extra Trees Parameters are: ', grid_extra_trees.best_params_)

print('The Best Extra Trees score is: ', grid_extra_trees.best_score_)
# Parameters for Light Gradient Boosting Machines Hyperparameter tuning

param_grid_lgbm = {'n_estimators': np.logspace(2,3.5,8).astype(int),

                   'feature_fraction': [0.1,0.3,0.5,0.7,0.9],

                   'bagging_fraction': [0.5,0.6,0.7,0.8,0.9],

                   'max_depth': np.logspace(0,3,10).astype(int),

                   'min_samples_split': [2, 5, 10],

                   'min_data_in_leaf': [1, 2, 4],

                   'learning_rate':[0.005,0.01,0.05,0.1,0.5]}



# Instantiate ExtraTreesClassifier

lgbm = LGBMClassifier(random_state = 42, is_provide_training_metric = True)



# Create a Random Search Parameter Grid

grid_lgbm = RandomizedSearchCV(estimator=lgbm, 

                          param_distributions=param_grid_lgbm, 

                          n_iter=100,

                          cv=3, 

                          verbose=3, 

                          n_jobs=1,

                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 

                          refit='NLL')

# Fit LightGBM Models - 100 models each with 3 rounds of cross validation - 300 fitted models

grid_lgbm.fit(X,y)

print('The Best LightGBM Estimator is: ', grid_lgbm.best_estimator_)

print('The Best LightGBM Parameters are: ', grid_lgbm.best_params_)

print('The Best LightGBM score is: ', grid_lgbm.best_score_)