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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
df.dropna(inplace=True)
df.drop(['Id','groupId','matchId','headshotKills','matchType','roadKills','vehicleDestroys','teamKills'],axis=1,inplace=True)
from sklearn import model_selection
x = df.drop('winPlacePerc',axis=1)

y = df['winPlacePerc']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.35,test_size=0.65, random_state=101)
x_train.head()
x_train.shape
import xgboost

from sklearn.model_selection import RandomizedSearchCV

classifier = xgboost.XGBRegressor()

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
#hyperparameter optimisation



n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
# Set up the random search with 4-fold cross validation

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=3, n_iter=4,

            scoring = 'neg_mean_absolute_error',

            verbose = 5,n_jobs = 4, 

            return_train_score = True,

            random_state=8)
random_cv.fit(x_train,y_train)
random_cv.best_estimator_
regressor = xgboost.XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.05, max_delta_step=0, max_depth=2,

             min_child_weight=2, missing=None, monotone_constraints='()',

             n_estimators=1500, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)
regressor.fit(x,y)
import pickle

filename = 'finalized_model.pkl'

pickle.dump(classifier, open(filename, 'wb'))
df_test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
df_test.drop(['Id','groupId','matchId','headshotKills','matchType','roadKills','vehicleDestroys','teamKills'],axis=1,inplace=True)
y_pred = regressor.predict(x_test)
y_pred
pred = pd.DataFrame(y_pred)

sub_df = pd.read_csv('../input/pubg-finish-placement-prediction/sample_submission_V2.csv')

datasets = pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','winPlacePerc']

datasets.to_csv('submission.csv',index=False)
import sklearn

mse = sklearn.metrics.mean_absolute_error(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
mse