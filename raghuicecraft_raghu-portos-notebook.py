"""

References or inspired by

1. https://www.kaggle.com/sudosudoohio/stratified-kfold-xgboost-eda-tutorial-0-281?scriptVersionId=1579846

2. https://www.kaggle.com/youhanlee/eda-stratifiedshufflesplit-xgboost-for-starter?scriptVersionId=1583200

"""

import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt   

import seaborn as sns             # not used



from sklearn import preprocessing 

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb



%matplotlib inline

warnings.filterwarnings('ignore')
# common gini metric for the gradient boost

# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897

def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score
# train and test data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(10)

test.head(10)
# pre-processing, null data

train_null = train.isnull().values.any()

print(train_null)



# features and drop related columns 

features = train.drop(['id','target'], axis=1).values

targets = train.target.values

drop_columns = train.columns[train.columns.str.startswith('ps_calc_')]

train = train.drop(drop_columns, axis=1)  

test = test.drop(drop_columns, axis=1)  
# model setting

kfold = 5

skf = StratifiedKFold(n_splits=kfold, random_state=99)

params = {

    'min_child_weight': 10.0,

    'objective': 'binary:logistic',

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 700

    }



X = train.drop(['id', 'target'], axis=1).values

y = train.target.values

test_id = test.id.values

test = test.drop('id', axis=1)
# submission preparation

sub = pd.DataFrame()

sub['id'] = test_id

sub['target'] = np.zeros_like(test_id)



for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print('[Fold %d/%d]' % (i + 1, kfold))

    X_train, X_valid = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    

    # Convert our data into XGBoost format

    d_train = xgb.DMatrix(X_train, y_train)

    d_valid = xgb.DMatrix(X_valid, y_valid)

    d_test = xgb.DMatrix(test.values)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]



    # Train the model! We pass in a max of 2,000 rounds (with early stopping after 100)

    # and the custom metric (maximize=True tells xgb that higher metric is better)

    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)



    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))

    # Predict on our test data

    p_test = mdl.predict(d_test)

    sub['target'] += p_test/kfold
# csv file

sub.to_csv('StratifiedKFold.csv', index=False)