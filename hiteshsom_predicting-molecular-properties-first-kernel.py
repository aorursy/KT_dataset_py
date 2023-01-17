#!pip install category_encoders

#!pip install LightGBM



import os

import gc

import math

import numpy as np

import pandas as pd

import category_encoders as ce

import lightgbm as lgbm

import matplotlib.pyplot as plt



from copy import copy

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error as mae
# Setting to display more columns than default



pd.set_option("display.max_rows", 1000)

gc.collect()
# Data Loading

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample_sub = pd.read_csv("../input/sample_submission.csv")

structures = pd.read_csv("../input/structures.csv")



# Original size

print(f"train.shape: {train.shape}")

print(f"test.shape: {test.shape}")
X_train = train.drop(columns=['scalar_coupling_constant']).copy()

y_train = train['scalar_coupling_constant'].copy()

X_test = test.copy()



X_train = X_train.drop(columns = ['id'])

X_test = X_test.drop(columns = ['id'])



# This is done because some steps in data pre-processing can chnage the order of rows in X_train

# and order of rows in target may not be chnaged becuase there is not processing done on it. So to maintain which row in X_train corresponds to 

# target the index is reset.

X_train = X_train.reset_index()

X_test = X_test.reset_index()
# function to convert object columns to category



def convert_object_to_categories(X_train, X_test):

    for col in X_train.columns:

        if X_train[col].dtype == 'O':

            X_train[col] = X_train[col].astype('category')

            X_test[col] = X_test[col].astype('category')

    return X_train, X_test



X_train, X_test = convert_object_to_categories(X_train, X_test)
# The score function



def calc_score(X_train, y_train, y_val):

    X_train_new = X_train.copy()

    y_train_new = y_train.copy()

    y_val_new = y_val.copy()

    y_val_new = pd.Series(y_val_new)

    X_train_new = X_train_new.reset_index(drop=True)

    y_train_new = y_train_new.reset_index(drop=True)

    X_train_new = X_train_new.merge(pd.DataFrame(y_train_new, columns=['scalar_coupling_constant']), left_index=True, right_index=True)

    X_train_new = X_train_new.merge(pd.DataFrame(y_val_new, columns=['y_val']), left_index=True, right_index=True)

    X_train_new['error'] = (X_train_new['scalar_coupling_constant'] - X_train_new['y_val']).abs()

    X_train_new['count'] = 1

    score_df = X_train_new.groupby(by = ['type']).agg({'count': 'count', 'error': 'sum'})

    score_df['error'] = (score_df['error']/score_df['count']).apply(np.log, dtype=float)

    score = (1/score_df.shape[0])*(score_df['error'].sum())

    return score
# Cross Validation



def cross_val(X, y):

    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    fold = 0

    for train_index, val_index in kf.split(X):

        fold +=1

        lgbm_model = lgbm.LGBMRegressor(random_state=10, n_estimators=1000)

        lgbm_model.fit(X.loc[train_index,:], y[train_index])

        y_val = lgbm_model.predict(X.loc[val_index,:])

        print(f"fold{fold} score: {calc_score(X.loc[val_index,:],y[val_index],y_val)}")
# Baseline cross validation score

# cross_val(X_train, y_train)



# score: 1.13 approx
# Baseline model

# lgbm_model = lgbm.LGBMRegressor()

# lgbm_model.fit(X_train, y_train)

# y_predict = lgbm_model.predict(X_test)
# Joining training/test and structures



X_train = X_train.merge(structures, left_on = ['molecule_name','atom_index_0'], right_on = ['molecule_name', 'atom_index'], sort = True)

X_train = X_train.rename(columns={'atom_index': 'atom_index_0_0', 'x':'atom_index_0_x', 'y':'atom_index_0_y', 'z':'atom_index_0_z', 'atom': 'atom_0'})

X_test = X_test.merge(structures, left_on = ['molecule_name','atom_index_0'], right_on = ['molecule_name', 'atom_index'], sort = True)

X_test = X_test.rename(columns={'atom_index': 'atom_index_0_0', 'x':'atom_index_0_x', 'y':'atom_index_0_y', 'z':'atom_index_0_z', 'atom': 'atom_0'})





X_train = X_train.merge(structures, left_on = ['molecule_name','atom_index_1'], right_on = ['molecule_name', 'atom_index'], sort = True)

X_train = X_train.rename(columns={'atom_index': 'atom_index_1_1', 'x':'atom_index_1_x', 'y':'atom_index_1_y', 'z':'atom_index_1_z', 'atom': 'atom_1'})

X_test = X_test.merge(structures, left_on = ['molecule_name','atom_index_1'], right_on = ['molecule_name', 'atom_index'], sort = True)

X_test = X_test.rename(columns={'atom_index': 'atom_index_1_1', 'x':'atom_index_1_x', 'y':'atom_index_1_y', 'z':'atom_index_1_z', 'atom': 'atom_1'})
# Dropping redundant columns



X_train = X_train.drop(columns=['atom_index_0_0','atom_index_1_1'])

X_test = X_test.drop(columns=['atom_index_0_0','atom_index_1_1'])
# Distance between atoms



X_train['distance'] = ((X_train['atom_index_0_x'] - X_train['atom_index_1_x'])**2 + (X_train['atom_index_0_y'] - X_train['atom_index_1_y'])**2 + (X_train['atom_index_0_z'] - X_train['atom_index_1_z'])**2)**0.5

X_test['distance'] = ((X_test['atom_index_0_x'] - X_test['atom_index_1_x'])**2 + (X_test['atom_index_0_y'] - X_test['atom_index_1_y'])**2 + (X_test['atom_index_0_z'] - X_test['atom_index_1_z'])**2)**0.5  
# Making column based in join type (eg. 2JHC is 2J type of join)



X_train['join_type'] = X_train['type'].str.slice(0,2)

X_test['join_type'] = X_test['type'].str.slice(0,2)
print(X_train['atom_0'].unique())

print(X_test['atom_0'].unique())



# So basically atom 0 in all molecules is always hydrogen. Hence will not hep in predicting target so we can drop it



X_train = X_train.drop(columns=['atom_0'])

X_test = X_test.drop(columns=['atom_0'])
# Number of atoms in molecule



X_train['num_atoms']=X_train.groupby(['molecule_name'])['atom_index_0'].transform('max') + 1

X_test['num_atoms']=X_test.groupby(['molecule_name'])['atom_index_0'].transform('max') + 1
# Lets check correlation between all the variables



df = X_train.set_index(keys='index', drop=False).merge(pd.DataFrame(y_train, columns=['scalar_coupling_constant']), left_index=True, right_index=True)

df.corr()
X_train, X_test = convert_object_to_categories(X_train, X_test)
# Setting index column as index so that cross validation function can pick matching rows from X_train_new and y_train. This is the reason 

# index was reset earlier. Reseting and setting index for test is not important would have made no chnage but I did because code looks more 

# consistent and also doing it is no harm



X_train_new = X_train.set_index(keys='index')

X_test_new = X_test.set_index(keys='index')
# cross_val(X_train_new, y_train)

# score is 0.7 approx
# So that dataframe looks fine.



X_train_new = X_train_new.sort_index(axis=0)

X_test_new = X_test_new.sort_index(axis=0)
# Making num_of_bonds in between atoms

# This is just an intuition 1J is H to direct carbon(ie one bond), 2J- 'HH' hydrogens attached to a common carbon atom so (2 bonds in between) or HC means hydrogen and carbvon but different carbon which is two bonds away   

# 3J 'HH' mean hydrgen attached to diferent carbons (which are 3 bonds away)



X_train_new['num_bonds'] = X_train_new['join_type'].str.slice(0,1)

X_test_new['num_bonds'] = X_test_new['join_type'].str.slice(0,1)



# 'num_bonds' should be integer

X_train_new['num_bonds'] = X_train_new['num_bonds'].astype('int')

X_test_new['num_bonds'] = X_test_new['num_bonds'].astype('int')
# Making angle between atoms, may be an important feature engineering. Angle will be in radians



def angle_between_vectors(df):

    dot_products = (df['atom_index_0_x']*df['atom_index_1_x'] + df['atom_index_0_y']*df['atom_index_1_y'] + df['atom_index_0_z']*df['atom_index_1_z'])

    magnitudes_product = (df['atom_index_0_x']**2+df['atom_index_0_y']**2+df['atom_index_0_z']**2)**0.5*(df['atom_index_1_x']**2+df['atom_index_1_y']**2+df['atom_index_1_z']**2)**0.5

    df['angle'] = np.arccos(dot_products/magnitudes_product)

    return df



X_train_new = angle_between_vectors(X_train_new)

X_test_new = angle_between_vectors(X_test_new)
# cross_val(X_train_new, y_train)



# score: 0.66 approx
# Training error



# lgbm_model = lgbm.LGBMRegressor(random_state=10, n_estimators=1000)

# lgbm_model.fit(X_train_new, y_train)

# y_predict = lgbm_model.predict(X_train_new)

# print(f"training score: {calc_score(X_train_new, y_train, y_predict)}")



# training score: 0.6372931683989951
# Modelling. I tried to see if increasing n_estomators improves score. It does so I am taking n_estomator=1000 

# More estimators also takes a lot of time to compute.



lgbm_model = lgbm.LGBMRegressor(random_state=10, n_estimators=1000)

lgbm_model.fit(X_train_new, y_train)

y_predict = lgbm_model.predict(X_test_new)
fig, ax = plt.subplots(figsize=(12,9))

lgbm.plot_importance(lgbm_model, ax)
sample_sub['scalar_coupling_constant'] = list(y_predict)

sample_sub.to_csv("submission.csv", index=False)