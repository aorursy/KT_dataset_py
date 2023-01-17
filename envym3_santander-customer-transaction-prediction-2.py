import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
PATH='../input/santander-customer-transaction-prediction/'

os.listdir(PATH)
%%time

train_df = pd.read_csv(PATH+"train.csv")

test_df = pd.read_csv(PATH+"test.csv")
train_df.shape, test_df.shape
train_df.target.value_counts(normalize=1)
%%time

idx = features = train_df.columns.values[2:202]

for df in [test_df, train_df]:

    df['sum'] = df[idx].sum(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)
features = train_df.columns.values[202:]
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

for feature in features:

    train_df['r2_'+feature] = np.round(train_df[feature], 2)

    test_df['r2_'+feature] = np.round(test_df[feature], 2)

    train_df['r1_'+feature] = np.round(train_df[feature], 1)

    test_df['r1_'+feature] = np.round(test_df[feature], 1)
print('Train and test columns: {} {}'.format(len(train_df.columns), len(test_df.columns)))
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

target = train_df['target']
param = {

        'num_leaves': 6,

        'max_bin': 63,

        'min_data_in_leaf': 45,

        'learning_rate': 0.01,

        'min_sum_hessian_in_leaf': 0.000446,

        'bagging_fraction': 0.55, 

        'bagging_freq': 5, 

        'max_depth': 14,

        'save_binary': True,

        'seed': 31452,

        'feature_fraction_seed': 31415,

         'feature_fraction': 0.51,

        'bagging_seed': 31415,

        'drop_seed': 31415,

        'data_random_seed': 31415,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

    }
from imblearn.over_sampling import SMOTE
#Using smote to handle imbalance

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(train_df.loc[:,features], train_df['target'])



X= pd.DataFrame(columns=features, data=X_res)

Y=pd.DataFrame(columns=['target'],data=y_res)
#Saving resampled data

train_df= pd.concat([X,Y],axis=1)
target= pd.Series(data=y_res)

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=31415)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 15000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
# cols = (feature_importance_df[["Feature", "importance"]]

#         .groupby("Feature")

#         .mean()

#         .sort_values(by="importance", ascending=False)[:150].index)

# best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



# plt.figure(figsize=(14,28))

# sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

# plt.title('Features importance (averaged/folds)')

# plt.tight_layout()

# plt.savefig('FI.png')
cols = (feature_importance_df.iloc[:1]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:150].index)
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission.csv", index=False)
!kaggle competitions submit -c santander-customer-transaction-prediction  -f /kaggle/working/submission.csv -m "first_sumb" 