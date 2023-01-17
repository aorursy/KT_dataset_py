import pandas as pd

import numpy as np

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import log_loss



import random

import os

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA



warnings.filterwarnings('ignore')

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline
%%time

files = ['../input/lish-moa/test_features.csv', 

         '../input/lish-moa/train_targets_scored.csv',

         '../input/lish-moa/train_features.csv',

         '../input/lish-moa/train_targets_nonscored.csv',

         '../input/lish-moa/sample_submission.csv']



def load_data(file):

    return pd.read_csv(file)



with multiprocessing.Pool() as pool:

    test, train_target, train, train_nonscored, sub = pool.map(load_data, files)

    

    

    

def mapping_and_filter(train, train_targets, test):

    cp_type = {'trt_cp': 0, 'ctl_vehicle': 1}

    cp_dose = {'D1': 0, 'D2': 1}

    for df in [train, test]:

        df['cp_type'] = df['cp_type'].map(cp_type)

        df['cp_dose'] = df['cp_dose'].map(cp_dose)

    train_targets = train_targets[train['cp_type'] == 0].reset_index(drop = True)

    train = train[train['cp_type'] == 0].reset_index(drop = True)

    train_targets.drop(['sig_id'], inplace = True, axis = 1)

    return train, train_targets, test



# Function to scale our data

def scaling(train, test):

    features = train.columns[2:]

    scaler = RobustScaler()

    scaler.fit(pd.concat([train[features], test[features]], axis = 0))

    train[features] = scaler.transform(train[features])

    test[features] = scaler.transform(test[features])

    return train, test, features



# Function to extract pca features

def fe_pca(train, test, n_components_g = 520, n_components_c = 46, SEED = 123):

    

    features_g = list(train.columns[4:776])

    features_c = list(train.columns[776:876])

    

    def create_pca(train, test, features, kind = 'g', n_components = n_components_g):

        train_ = train[features].copy()

        test_ = test[features].copy()

        data = pd.concat([train_, test_], axis = 0)

        pca = PCA(n_components = n_components,  random_state = SEED)

        data = pca.fit_transform(data)

        columns = [f'pca_{kind}{i + 1}' for i in range(n_components)]

        data = pd.DataFrame(data, columns = columns)

        train_ = data.iloc[:train.shape[0]]

        test_ = data.iloc[train.shape[0]:].reset_index(drop = True)

        train = pd.concat([train, train_], axis = 1)

        test = pd.concat([test, test_], axis = 1)

        return train, test

    

    train, test = create_pca(train, test, features_g, kind = 'g', n_components = n_components_g)

    train, test = create_pca(train, test, features_c, kind = 'c', n_components = n_components_c)

    return train, test



# Function to extract common stats features

def fe_stats(train, test):

    

    features_g = list(train.columns[4:776])

    features_c = list(train.columns[776:876])

    

    for df in [train, test]:

        df['g_sum'] = df[features_g].sum(axis = 1)

        df['g_mean'] = df[features_g].mean(axis = 1)

        df['g_std'] = df[features_g].std(axis = 1)

        df['g_kurt'] = df[features_g].kurtosis(axis = 1)

        df['g_skew'] = df[features_g].skew(axis = 1)

        df['c_sum'] = df[features_c].sum(axis = 1)

        df['c_mean'] = df[features_c].mean(axis = 1)

        df['c_std'] = df[features_c].std(axis = 1)

        df['c_kurt'] = df[features_c].kurtosis(axis = 1)

        df['c_skew'] = df[features_c].skew(axis = 1)

        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)

        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)

        df['gc_std'] = df[features_g + features_c].std(axis = 1)

        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)

        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)

        

    return train, test



def c_squared(train, test):

    

    features_c = list(train.columns[776:876])

    for df in [train, test]:

        for feature in features_c:

            df[f'{feature}_squared'] = df[feature] ** 2

    return train, test



# Function to calculate the mean log loss of the targets including clipping

def mean_log_loss(y_true, y_pred):

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    metrics = []

    for target in range(206):

        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))

    return np.mean(metrics)
print(train.shape)

print(test.shape)
#train = pd.read_csv('../input/lish-moa/train_features.csv')

#train_target = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

#test = pd.read_csv('../input/lish-moa/test_features.csv')

#sub = pd.read_csv('../input/lish-moa/sample_submission.csv')

train, train_target, test = mapping_and_filter(train, train_target, test)

train, test = fe_stats(train, test)

train, test = c_squared(train, test)

train, test = fe_pca(train, test, n_components_g = 520, n_components_c = 46, SEED = 123)

train, test, features = scaling(train, test)
print(train.shape)

print(test.shape)
targets = [col for col in train_target.columns if col != 'sig_id']

print('Number of different labels:', len(targets))
features = [col for col in train.columns if col != 'sig_id']

print('Number of features:', len(features))
print(train_target.shape)

#for feature in ['cp_type', 'cp_dose']:

#    le = LabelEncoder()

#    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

#    train[feature] = le.transform(list(train[feature].astype(str).values))

#    test[feature] = le.transform(list(test[feature].astype(str).values))
X = train[features]
print(X.shape)
params = {'num_leaves': 491,

          'min_child_weight': 0.03,

          'feature_fraction': 0.3,

          'bagging_fraction': 0.4,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'binary_logloss',

          "verbosity": 0,

          'reg_alpha': 0.4,

          'reg_lambda': 0.6,

          'random_state': 47

         }
accumulative_loss = 0

skf = StratifiedKFold(n_splits=3, random_state=47, shuffle=True)



# 206 different models. One for each label

for model, target in enumerate(targets, 1):

    y = train_target[target]

    start_time = time()

    preds = np.zeros(test.shape[0])

    oof = np.zeros(X.shape[0])



    for trn_idx, test_idx in skf.split(X, y):

        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

        val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])

        clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds=25)

        oof[test_idx] = clf.predict(X.iloc[test_idx])

        preds += clf.predict(test[features]) / skf.n_splits



    sub[target] = preds

    loss = log_loss(y, oof)

    accumulative_loss += loss

    print('[{}] Model: {} logloss: {:.3f}'.format(str(datetime.timedelta(seconds=time() - start_time))[:7], model, loss))



    del preds, oof, start_time, y, loss

    gc.collect();
print('Overall mean loss: {:.3f}'.format(accumulative_loss / 206))
def submission(test_pred):

    sub.loc[:, train_target.columns] = test_pred

    sub.loc[test['cp_type'] == 1, train_target.columns] = 0

    sub.to_csv('submission.csv', index = False)

    return sub
#submission(sub)
sub = submission(sub)

sub.head()
# sub.to_csv('submission.csv', index=False)