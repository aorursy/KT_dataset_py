import pandas as pd

import numpy as np

import gc



# Gradient Boosting

import lightgbm as lgb

import xgboost as xgb



# Scikit-learn

from sklearn.metrics import average_precision_score

from sklearn.model_selection import StratifiedKFold 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc



# Graphics

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Skopt functions

from skopt import BayesSearchCV

from skopt import gp_minimize # Bayesian optimization using Gaussian Processes

from skopt.space import Real, Categorical, Integer

from skopt.utils import use_named_args # decorator to convert a list of parameters to named arguments

from skopt.callbacks import DeadlineStopper # Stop the optimization before running out of a fixed budget of time.

from skopt.callbacks import VerboseCallback # Callback to control the verbosity

from skopt.callbacks import DeltaXStopper # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta



# Hyperparameters distributions

from scipy.stats import randint

from scipy.stats import uniform



# Metrics

from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error



import os

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn
santander_data = pd.read_csv('../input/train.csv')

santander_data_test = pd.read_csv('../input/test.csv')
# Taking the labels (price)

label_df = santander_data['target']
santander_data.drop(['ID_code','target'], axis=1, inplace=True)



santander_data_test.drop('ID_code', axis=1, inplace=True)

santander_data.head(10)
santander_data_test.head(10)
santander_data.describe()
santander_data[santander_data.isnull().any(axis=1)]
santander_data.select_dtypes(exclude=np.number).columns
len_train = len(santander_data)

len_train
#Merge test and train

merged = pd.concat([santander_data, santander_data_test])

#Saving the list of original features in a new list `original_features`.

original_features = merged.columns

merged.shape
idx = features = merged.columns.values[0:200]

for df in [merged]:

    df['sum'] = df[idx].sum(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)
print("Total number of features: ",merged.shape[1])
train_df = merged.iloc[:len_train]

train_df.head()
X_test = merged.iloc[len_train:]

X_test.head()
def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
"""train_df = santander_data

X_test = santander_data_test"""

del santander_data

del santander_data_test

gc.collect()
skf_three= StratifiedKFold(n_splits=7, shuffle=False, random_state=2319)
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.335,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.041,

    'learning_rate': 0.0083,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': -1

}
# Create arrays and dataframes to store results

oof_preds = np.zeros(train_df.shape[0])

sub_preds = np.zeros(len(X_test))

feats = [f for f in train_df.columns]

    

for n_fold, (train_idx, valid_idx) in enumerate(skf_three.split(train_df[feats], label_df)):

    X_train, y_train = train_df.iloc[train_idx][feats], label_df.iloc[train_idx]

    X_valid, y_valid = train_df.iloc[valid_idx][feats], label_df.iloc[valid_idx]

    

    X_tr, y_tr = augment(X_train.values, y_train.values)

    X_tr = pd.DataFrame(X_tr)

    

    print("Fold idx:{}".format(n_fold + 1))

    trn_data = lgb.Dataset(X_tr, label=y_tr)

    val_data = lgb.Dataset(X_valid, label=y_valid)

        

    clf = lgb.train(param, trn_data,1000000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)

        



    oof_preds[valid_idx] = clf.predict(train_df.iloc[valid_idx][feats], num_iteration=clf.best_iteration)

    sub_preds += clf.predict(X_test[feats], num_iteration=clf.best_iteration) / 7





print('Full AUC score %.6f' % roc_auc_score(label_df, oof_preds))



pred3=sub_preds
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['target'] = pred3

sample_submission.to_csv('submission.csv', index=False)