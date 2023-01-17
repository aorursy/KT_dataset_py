# import the libraries

import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import GridSearchCV



%matplotlib inline
# read the data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# store the target variable

y_train_all = df_train.y

id_test = df_test.ID



# drop the target variable and the ids before the combination

df_train.drop(['ID', 'y'], axis=1, inplace=True)

df_test.drop(['ID'], axis=1, inplace=True)



# Build df_all = (df_train+df_test)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

print(df_all.shape)
# drop the 30 lowest XGB scored features

lowest_scored_thirty = ['X344', 'X20','X117','X109','X378','X45','X362','X161','X164','X61',

 'X65','X380','X154', 'X300','X77', 'X114', 'X85', 'X321', 'X195','X209', 'X206', 'X283', 'X343', 'X340', 'X376',

 'X36', 'X375', 'X264', 'X250', 'X329']

df_all = df_all.drop(lowest_scored_thirty, axis=1)

df_all.shape
# these are the features that were eliminated using LassoCV

lasso_eliminated_features = ['X3', 'X0', 'X314', 'X350', 'X315', 'X180', 'X27', 'X261', 

                             'X220', 'X321', 'X355', 'X29', 'X136']



to_eliminate = list(set(lasso_eliminated_features) - set(lowest_scored_thirty))



df_all = df_all.drop(to_eliminate, axis=1)

df_all.shape
# factorize the categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()

print(df_obj.shape)

print(df_numeric.shape)



# drop the numeric features where the column contains only one unique value

for col in df_numeric:

    cardinality = len(np.unique(df_train[col]))

    if cardinality == 1:

        df_numeric = df_numeric.drop(col, axis=1)

        

for col in df_obj:

    df_obj[col] = pd.factorize(df_obj[col])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)

print(df_values.shape)
# now convert to numpy values

X_all = df_values.values

print(X_all.shape)



# the complete training set

X_train_all = X_all[:num_train]



# create the validation sets

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=4242)



X_test = X_all[num_train:]



df_columns = df_values.columns
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=3)
from sklearn.linear_model import ElasticNetCV

model_elastic = ElasticNetCV(l1_ratio=[.1, .4, .5, .6, .7, .8, .9, .95, .99, 1], cv=5)
from sklearn.linear_model import LassoCV

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0001, 0.0005], cv=5)
# the meta regressor

from sklearn.ensemble import ExtraTreesRegressor

tree_model = ExtraTreesRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=5)
# this is the stacking part

from mlxtend.regressor import StackingRegressor

stregr = StackingRegressor(regressors=[random_forest, model_lasso, model_elastic], 

                           meta_regressor=tree_model)

# fit the stacked model

stregr.fit(X_train, y_train)
from sklearn.metrics import r2_score



# check the r2 value of the stacked model predictions

predict_val = stregr.predict(X_val)



r2_score(y_val, predict_val)