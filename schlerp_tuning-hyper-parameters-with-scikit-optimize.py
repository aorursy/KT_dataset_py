import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))

train_df = pd.read_csv('../input/train.csv')

# borrowed from other kagglers: https://www.kaggle.com/hmendonca/testing-engineered-features-lb-1-42
# Find and drop duplicate rows
t = train_df.iloc[:,2:].duplicated(keep=False)
duplicated_indices = t[t].index.values
print("Removed {} duplicated rows: {}".format(len(duplicated_indices), duplicated_indices))
train_df.iat[duplicated_indices[0], 1] = np.expm1(np.log1p(train_df.target.loc[duplicated_indices]).mean()) # keep and update first with log mean
train_df.drop(duplicated_indices[1:], inplace=True) # drop remaining

# Columns to drop because there is no variation in training set
zero_std_cols = train_df.drop("ID", axis=1).columns[train_df.std() == 0]
train_df.drop(zero_std_cols, axis=1, inplace=True)
print("Removed {} constant columns".format(len(zero_std_cols)))
# Log-transform all column
train_df = train_df.drop(['ID'], axis=1)
train_df = np.log1p(train_df)
X = train_df.drop(['target'], axis=1)
Y = train_df[['target']]

# we will use only a subset of the train data so this kaggle kerenel will run quicker...
from sklearn.model_selection import train_test_split
X, _, Y, __ = train_test_split(X, Y, test_size=0.33)
Y = np.array(Y).reshape(len(Y))
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize

# set up hyperparameter space
space = [Integer(16, 256, name='num_leaves'),
         Integer(8, 256, name='n_estimators'),
         Categorical(['gbdt', 'dart', 'goss'], name='boosting_type'),
         Real(0.001, 1.0, name='learning_rate')]

import lightgbm
regressor = lightgbm.LGBMRegressor()

from sklearn.model_selection import cross_val_score

@use_named_args(space)
def objective(**params):
    regressor.set_params(**params)
    return -np.mean(cross_val_score(regressor, X, Y, cv=5, n_jobs=1, scoring='neg_mean_absolute_error'))
reg_gp = gp_minimize(objective, space, verbose=True)

print('best score: {}'.format(reg_gp.fun))

print('best params:')
print('       num_leaves: {}'.format(reg_gp.x[0]))
print('     n_estimators: {}'.format(reg_gp.x[1]))
print('    boosting_type: {}'.format(reg_gp.x[2]))
print('    learning_rate: {}'.format(reg_gp.x[3]))