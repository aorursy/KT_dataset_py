import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb # xgboost package 

import gc # to take out da trash [memory management]
# Lets see whats in our input folder

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Load Data

train = pd.read_csv('../input/train_2016_v2.csv')

prop = pd.read_csv('../input/properties_2016.csv')

sample = pd.read_csv('../input/sample_submission.csv')
# Print Data Shape

# In order to submit to Kaggle we'll be modifiying the sample dataset with our predictions

print (train.shape, prop.shape, sample.shape)
# Convert to Float32 

# This is so that our script can run on Kaggle Kernels

# Kaggle has a memory limit on the Kernels so this is a necessary step

# We're turning 64 bit floats into 32 bit floats



for c, dtype in zip(prop.columns, prop.dtypes):

    if dtype == np.float64:

        prop[c] = prop[c].astype(np.float32)
# Merge training dataset with properties dataset

df_train = train.merge(prop, how='left', on='parcelid')



# Remove useless columns (anything used for ID purposes, has no variation, or is not suitable for training)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)



# Save Train columns

train_columns = x_train.columns



# Train our model to predict log error

y_train = df_train['logerror'].values



# Binarify our categorical column variables to remove NaN objects

for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)



# Delete our old training dataset; take out the trash

del df_train; gc.collect()
# Split dataset at roughly the ~88% mark into training and validation datasets

# We'll be evaluating the fine tuning of our model by seeing how it runs on the validation dataset



split = 80000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
# Split training and validation datasets



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



del x_train, x_valid; gc.collect()
# Set hyperparameters

# Only hyperparamater that is relevant for optimizing here (in Anokas's notebook) is max_depth.  

# 

# When I build my own submission I will try tuning gamma, min_child_weight, subsample, 

# colsample_bytree, as well as the regularization paramaters.



params = {}

params['eta'] = 0.02 # control the learning rate: scale the contribution of each tree by a factor of 0 < eta < 1. Lower is slower but more robust to overfitting.

params['objective'] = 'reg:linear' # Default.  Running a regression, since we're predicting values not classes

params['eval_metric'] = 'mae' # We're evaluating our models on Mean Average Error.  

params['max_depth'] = 4 # Maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting.

params['silent'] = 1 # Don't print messages



# Train model

#

# 'Watchlist' is an evaluation dataset- We will be tuning our model based on how it does in the validation dataset 

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#

# Anokas has implemented early stopping.  Once we reach the point where our validation score no 

# longer improves after a set number of iterations (100 in this case) we use the model run that preceeded the 

# chain of 100 un-changed iterations.  This is to prevent additional overfitting where it does not improve the model.

clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)
# Build test set



sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(prop, on='parcelid', how='left')



# Memory Management

del prop; gc.collect()



# Binarify the data (ie remove NaN, set it to False)

x_test = df_test[train_columns]

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)



# Memory management

del df_test, sample; gc.collect()



# Convert table to xgb format

d_test = xgb.DMatrix(x_test)



# Memory management

del x_test; gc.collect()
# Make predictions on data

p_test = clf.predict(d_test)



# Delete testset; take out trash

del d_test; gc.collect()



# Read sample subgmission

sub = pd.read_csv('../input/sample_submission.csv')

for c in sub.columns[sub.columns != 'ParcelId']:

    sub[c] = p_test



# Write submission

sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f')