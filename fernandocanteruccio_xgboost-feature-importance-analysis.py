# Import all the necessary packages 

import kagglegym

import numpy as np

import pandas as pd

import time

import xgboost as xgb

import matplotlib.pyplot as plt



%matplotlib inline
# Read the full data set stored as HDF5 file

train = pd.read_hdf('../input/train.h5')
# Observed with histograms:

low_y_cut = -0.086093

high_y_cut = 0.093497



y_is_above_cut = (train.y > high_y_cut)

y_is_below_cut = (train.y < low_y_cut)

y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
excl = ['id', 'sample', 'y', 'timestamp']

cols = [c for c in train.columns if c not in excl]

target_var = 'y'



features = train.loc[y_is_within_cut, cols]

targets = train.loc[y_is_within_cut, target_var]



X_train = features[train.timestamp <= 905].values

y_train = targets[train.timestamp <= 905].values

X_valid = features[train.timestamp > 905].values

y_valid = targets[train.timestamp > 905].values

feature_names = features.columns

del features, train, targets, cols, excl, target_var
print("train features shape:",X_train.shape)

print("train label shape:",y_train.shape)

print("validation features shape:",X_valid.shape)

print("validation labels shape:",y_valid.shape)

print("feature_names shape:", feature_names.shape)
xgmat_train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

xgmat_valid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
params_xgb = {'objective':'reg:linear',

              'eta'             : 0.1,

              'max_depth'       : 4,

              'subsample'       : 0.9,

              'min_child_weight': 1000,

              'base_score':0

              }
print ("Training")

t0 = time.time()

bst = xgb.train(params_xgb, xgmat_train, 10)

print("Done: %.1fs" % (time.time() - t0))
params_xgb.update({'process_type': 'update',

                   'updater'     : 'refresh',

                   'refresh_leaf': False})
t0 = time.time()

print("Refreshing")

bst_after = xgb.train(params_xgb, xgmat_valid, 10, xgb_model=bst)

print("Done: %.1fs" % (time.time() - t0))
imp = pd.DataFrame(index=feature_names)

imp['train'] = pd.Series(bst.get_score(importance_type='gain'), index=feature_names)



# OOB feature importance

imp['OOB'] = pd.Series(bst_after.get_score(importance_type='gain'), index=feature_names)

imp = imp.fillna(0)
ax = imp.sort_values('train').tail(10).plot.barh(title='Feature importances sorted by train', figsize=(10,6))
ax = imp.sort_values('OOB').tail(10).plot.barh(title='Feature importances sorted by OOB', xlim=(0,0.04), figsize=(10,6))