!nvidia-smi
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import os 

import sys

import matplotlib.pyplot as plt

from pandas import plotting

from sklearn import preprocessing

import sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import f1_score

from sklearn.model_selection import KFold, train_test_split

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn import tree

from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

from sklearn.metrics import log_loss

from sklearn.multioutput import MultiOutputClassifier

from sklearn.preprocessing import RobustScaler

import pandas_profiling

from catboost import CatBoostClassifier

from catboost import cv

from catboost import Pool





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

trainTargets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

trainTargetsNonScored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

combinedTrain = train.merge(trainTargets)
featureCols = train.columns

labelCols = trainTargets.drop('sig_id',axis=1).columns
# shuffle the dataset

combinedTrain = combinedTrain.sample(frac=1, random_state= 2)

# delete the index

combinedTrain.reset_index(drop=True, inplace=True)

# show the head of combined df

combinedTrain.head()

# reconstruct the dataframe 

train = combinedTrain[featureCols]

trainTargets = combinedTrain[labelCols]
#check the cleanliness of the data 

train.isnull().sum().sum()
#step 1 - One hot Encoding

forEncoding = ['cp_time', 'cp_type', 'cp_dose']

train = pd.concat([train,(pd.get_dummies(train[forEncoding].astype(str),dummy_na=False, dtype=np.uint8,prefix="feature"))],axis=1)

test = pd.concat([test,(pd.get_dummies(test[forEncoding].astype(str),dummy_na=False, dtype=np.uint8,prefix="feature"))],axis=1)
# Step 2 - Get columns of features and labels

labelCols = [col for col in trainTargets.columns if col != 'sig_id']

featureCols = [col for col in train.columns if col not in ['sig_id', 'cp_time', 'cp_type', 'cp_dose']]

print('Number of different labels:', len(labelCols))

print('Number of features:', len(featureCols))
# Step 3 - Separate the Dataset

X = train[featureCols]

X_test = test[featureCols]

y = trainTargets
cat_features= ['feature_24',

 'feature_48',

 'feature_72',

 'feature_ctl_vehicle',

 'feature_trt_cp',

 'feature_D1',

 'feature_D2']

nonCat = [col for col in X.columns if col not in cat_features]
# Step 4 - Standardize the data using a robust scalar

rsc = RobustScaler()

X[nonCat] = rsc.fit_transform(X[nonCat])

X_test[nonCat] = rsc.transform(X_test[nonCat])

X = pd.DataFrame(X)

X_test = pd.DataFrame(X_test)
X
# Step 5 - Convert to numpy array

X = X.iloc[:].to_numpy()

X_test = X_test.iloc[:].to_numpy()

y = y.iloc[:].to_numpy()
print(X.shape)

print(X_test.shape)

print(y.shape)
# very initial model

xgb = XGBClassifier(n_estimators=500,seed=123,learning_rate=0.15,max_depth=5,colsample_bytree=1,subsample=1,tree_method='gpu_hist')

classifier = MultiOutputClassifier(xgb)
# Parameters from https://www.kaggle.com/fchmiel/xgboost-baseline-multilabel-classification

params = {'estimator__colsample_bytree': 0.6522,

          'estimator__gamma': 3.6975,

          'estimator__learning_rate': 0.0503,

          'estimator__max_delta_step': 2.0706,

          'estimator__max_depth': 10,

          'estimator__min_child_weight': 31.5800,

          'estimator__n_estimators': 166,

          'estimator__subsample': 0.8639

         }



classifier.set_params(**params)
# solution inspired from := https://www.kaggle.com/fchmiel/xgboost-baseline-multilabel-classification/log

test_preds = np.zeros((X_test.shape[0], y.shape[1]))

loss1 = []

kf=KFold(n_splits=5, random_state=100, shuffle=True)

for iteration, (train_index,validation_index) in enumerate(kf.split(X, y)):

    print('ITERATION NUMBER - ', iteration)

    X_train, X_val = X[train_index], X[validation_index]

    y_train, y_val = y[train_index], y[validation_index]

    

    classifier.fit(X_train, y_train)

    val_preds = classifier.predict_proba(X_val) 

    val_preds = np.array(val_preds)[:,:,1].T #(num_labels,num_samples,prob_0/1)

    

    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

    loss1.append(loss)

    preds = classifier.predict_proba(X_test)

    preds = np.array(preds)[:,:,1].T #(num_labels,num_samples,prob_0/1)

    test_preds += preds / 5 #take average of 10 models

    

print(loss1)

print('Mean CV loss across folds', np.mean(loss1))
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

trainTargets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

trainTargetsNonScored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

combinedTrain = train.merge(trainTargets)

featureCols = train.columns

labelCols = trainTargets.drop('sig_id',axis=1).columns
# shuffle the dataset

combinedTrain = combinedTrain.sample(frac=1, random_state= 2)

# delete the index

combinedTrain.reset_index(drop=True, inplace=True)

# show the head of combined df

combinedTrain.head()

# reconstruct the dataframe 

train = combinedTrain[featureCols]

trainTargets = combinedTrain[labelCols]
# Step 2 - Get columns of features and labels

labelCols = [col for col in trainTargets.columns if col != 'sig_id']

featureCols = [col for col in train.columns if col !='sig_id']

print('Number of different labels:', len(labelCols))

print('Number of features:', len(featureCols))
lEncode = preprocessing.LabelEncoder()

forEncoding = ['cp_time','cp_type','cp_dose']

# for train



for i in forEncoding:

  lEncode.fit(train[i])

  x = lEncode.transform(train[i])

  train[i] = x



# for test

for i in forEncoding:

  lEncode.fit(test[i])

  x = lEncode.transform(test[i])

  test[i] = x
# Step 3 - Separate the Dataset

X = train[featureCols]

X_test = test[featureCols]

y = trainTargets
cat_features= ['cp_time', 'cp_type', 'cp_dose']

nonCat = [col for col in X.columns if col not in cat_features]
# Step 4 - Standardize the data using a robust scalar

rsc = RobustScaler()

X[nonCat] = rsc.fit_transform(X[nonCat])

X_test[nonCat] = rsc.transform(X_test[nonCat])

X = pd.DataFrame(X)

X_test = pd.DataFrame(X_test)
# Step 5 - Convert to numpy array

X = X.iloc[:].to_numpy()

X_test = X_test.iloc[:].to_numpy()

y = y.iloc[:].to_numpy()
y.shape
cat_features= ['cp_time', 'cp_type', 'cp_dose']
# second model - Catboost

params = {'loss_function':'MultiClass',

          #'eval_metric':'los',

          #'cat_features': cat_features,

          'task_type': 'GPU',

          'border_count': 32,

          'verbose': 200,

          'random_seed': 1,

          #'learning_rate' : 0.1,

          #'random_strength' : 0.1,

          #'depth' : 8,

          'early_stopping_rounds' : 200,

          #'leaf_estimation_method' : 'Newton'

         }

cbc = CatBoostClassifier(**params)

classifier = MultiOutputClassifier(cbc)
classifier.fit(X, y)

preds = classifier.predict_proba(X_test)

preds = np.array(preds)[:,:,1].T

test_preds = (test_preds/2)  + (preds/2)
test_preds
#classifier.fit(train_data, # instead of X_train, y_train

#          eval_set=valid_data, # instead of (X_valid, y_valid)

#          use_best_model=True, 

#          plot=True

#         );
'''loss1 = []

kf=KFold(n_splits=3, random_state=100, shuffle=True)

for iteration, (train_index,validation_index) in enumerate(kf.split(X, y)):

    print('ITERATION NUMBER - ', iteration)

    X_train, X_val = X[train_index], X[validation_index]

    y_train, y_val = y[train_index], y[validation_index]

    

    classifier.fit(X_train, y_train)

    val_preds = classifier.predict_proba(X_val) 

    val_preds = np.array(val_preds)[:,1].T #(num_labels,num_samples,prob_0/1)

    

    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

    loss1.append(loss)

    preds = classifier.predict_proba(X_test)

    print(preds)

    preds = np.array(preds)[:,1].T #(num_labels,num_samples,prob_0/1)

    test_preds += preds / 3 #take average of 10 models

    

print(loss1)

print('Mean CV loss across folds', np.mean(loss1))'''
## from discussion at https://www.kaggle.com/c/lish-moa/discussion/180304

## and code from https://www.kaggle.com/fchmiel/xgboost-baseline-multilabel-classification/comments



# set control test preds to 0

mask = test['cp_type']=='ctl_vehicle' # wherever the cp_type = 'ctl_vehicle' 



test_preds[mask] = 0
# read the sample submission and make a new DF

sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

#sub = sub.iloc[:]

#sub = sub.to_frame()

sub.iloc[:,1:] = test_preds
sub.to_csv('submission.csv', index=False)