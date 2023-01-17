import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

import xgboost as xgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold, KFold

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
i = 1

for df in [test, train]:

    idx = df.columns.values[i:i+200]

    df['sum']  = df[idx].sum(axis=1)  

    df['min']  = df[idx].min(axis=1)

    df['max']  = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std']  = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med']  = df[idx].median(axis=1)

    i = i + 1
# thank you 

# https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment



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
#train = santander_data

#X_test = santander_data_test

#del santander_data

#del santander_data_test

#gc.collect()
features = [c for c in train.columns if c not in ['ID_code', 'target']]

target = train['target']
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.4,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 28,

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1}
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=31415)

oof = np.zeros(len(train))

predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 15000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 200)

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:150].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,28))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('Features importance (averaged/folds)')

plt.tight_layout()

plt.savefig('FI.png')
sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission.csv", index=False)
## Catboost : https://www.kaggle.com/wakamezake/starter-code-catboost-baseline

from catboost import Pool, CatBoostClassifier

model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC")

kf = KFold(n_splits=10, random_state=42, shuffle=True)



y_valid_pred = 0 * target

y_test_pred = 0



for idx, (train_index, valid_index) in enumerate(kf.split(train)):

    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]

    X_train, X_valid = train[features].iloc[train_index,:], train[features].iloc[valid_index,:]

    _train = Pool(X_train, label=y_train)

    _valid = Pool(X_valid, label=y_valid)

    print( "\nFold ", idx)

    fit_model = model.fit(_train,

                          eval_set=_valid,

                          use_best_model=True,

                          verbose=200

                         )

    pred = fit_model.predict_proba(X_valid)[:,1]

    print( "  auc = ", roc_auc_score(y_valid, pred) )

    y_valid_pred.iloc[valid_index] = pred

    y_test_pred += fit_model.predict_proba(test[features])[:,1]

y_test_pred /= 5
##submission

sub_df1 = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df1["target"] = y_test_pred

sub_df1.to_csv("cat_submission.csv", index=False)
sub_df2 = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df2["target"] = 0.5*sub_df["target"] + 0.5*sub_df1["target"]

sub_df2.to_csv("lgb_cat_submission.csv", index=False)