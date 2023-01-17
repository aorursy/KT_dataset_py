import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

import random

import datetime as dt

import gc

import seaborn as sns #python visualization library

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



color = sns.color_palette()

np.random.seed(1)
# dataset path

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/zillow-prize-1/train_2016_v2.csv', parse_dates=['transactiondate'])

properties = pd.read_csv('/kaggle/input/zillow-prize-1/properties_2016.csv')

test = pd.read_csv('/kaggle/input/zillow-prize-1/sample_submission.csv')

test= test.rename(columns={'ParcelId': 'parcelid'})  # To make it easier for merging datasets on same column_id later



print(f'train.shape: {train.shape}')

print(f'properties.shape: {properties.shape}')

print(f'test.shape: {test.shape}')
# convert properties df

for c, dtype in zip(properties.columns, properties.dtypes):

    if dtype == np.float64:

        properties[c] = properties[c].astype(np.float32)  # np.float64 -> np.float32

    if dtype == np.int64:

        properties[c] = properties[c].astype(np.int32)  # np.int64 -> np.int32



# convert test df

for column in test.columns:

    if test[column].dtype == int:

        test[column] = test[column].astype(np.int32)  # int -> np.int32

    if test[column].dtype == float:

        test[column] = test[column].astype(np.float32)  # float -> np.float32
# New Features

# living area proportions

properties['living_area_prop'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']

# tax value ratio

properties['value_ratio'] = properties['taxvaluedollarcnt'] / properties['taxamount']

# tax value proportions

properties['value_prop'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']
# Merging the Datasets

# We are merging the properties dataset with training and testing dataset for model building and testing prediction

df_train = train.merge(properties, how='left', on='parcelid')

df_test = test.merge(properties, how='left', on='parcelid')



# Remove previos variables to keep some memory

del properties, train

gc.collect()

print('Memory usage reduction…')
df_train.head()
df_test.head()
# some scaling

df_train[['latitude', 'longitude']] /= 1e6

df_test[['latitude', 'longitude']] /= 1e6

df_train['censustractandblock'] /= 1e12

df_test['censustractandblock'] /= 1e12
# Label Encoding

lbl = LabelEncoder()



# encoding df_train

for c in df_train.columns:

    df_train[c]=df_train[c].fillna(0)

    if df_train[c].dtype == 'object':

        lbl.fit(list(df_train[c].values))

        df_train[c] = lbl.transform(list(df_train[c].values))

        

# encoding df_test        

for c in df_test.columns:

    df_test[c]=df_test[c].fillna(0)

    if df_test[c].dtype == 'object':

        lbl.fit(list(df_test[c].values))

        df_test[c] = lbl.transform(list(df_test[c].values))
df_train.head()
df_test.head()
# Rearranging the DataSets

# We will now drop the features that serve no useful purpose

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

x_test = df_test.drop(['parcelid', 'propertyzoningdesc', 'propertycountylandusecode', '201610', '201611', '201612', '201710', '201711', '201712'], axis = 1)



print(f'x_train.shape: {x_train.shape}')

print(f'x_test.shape: {x_test.shape}')
# split dataset for cross validation

x_train = x_train.values

y_train = df_train['logerror'].values

X = x_train

y = y_train

Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.2, random_state=42)



print(f'Xtrain.shape: {Xtrain.shape}')

print(f'Xvalid.shape: {Xvalid.shape}')

print(f'ytrain.shape: {ytrain.shape}')

print(f'yvalid.shape: {yvalid.shape}')
# We can now select the parameters for Xgboost and monitor the progress of results on our validation set.

dtrain = xgb.DMatrix(Xtrain, label=ytrain)

dvalid = xgb.DMatrix(Xvalid, label=yvalid)

dtest = xgb.DMatrix(x_test.values)
type(dtrain)
# Try different parameters

xgb_params = {

    'min_child_weight': 5, 'eta': 0.035, 'colsample_bytree': 0.5, 'max_depth': 4,

    'subsample': 0.85, 'lambda': 0.8, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,

    'eval_metric': 'mae', 'objective': 'reg:linear'

}

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Train

model_xgb = xgb.train(xgb_params, dtrain, 1000, watchlist, early_stopping_rounds=100,maximize=False, verbose_eval=10)
# Predicting the results

Predicted_test_xgb = model_xgb.predict(dtest)  # ndarray

print(f'Predicted_test_xgb.shape: {Predicted_test_xgb}')
# Submitting the Results

sample_file = pd.read_csv('/kaggle/input/zillow-prize-1/sample_submission.csv')

print(f'sample_file.shape: {sample_file.shape}')
sample_file.head(2)
# Submitting the Results

for c in sample_file.columns[sample_file.columns != 'ParcelId']:

    sample_file[c] = Predicted_test_xgb

    print('Preparing the csv file …')

    

# write csv file

sample_file.to_csv('xgb_predicted_results.csv', index=False, float_format='%.4f')

print('Finished writing the file')