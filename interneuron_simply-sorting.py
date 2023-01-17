import numpy as np 

import pandas as pd 

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
ftz = ['var_41', 'var_30', 'var_185', 

       'var_27', 'var_103', 'var_100', 

       'var_38', 'var_17', 'var_161', 'var_99', 'var_184', 'var_188']
param = {'bagging_freq': 5,

         'bagging_fraction': 0.331,

         'boost_from_average':'false',

         'boost': 'gbdt',

         'feature_fraction': 0.0405,

         'learning_rate': 0.0083,

         'max_depth': -1,

         'metric':'auc',

         'min_data_in_leaf': 80,

         'min_sum_hessian_in_leaf': 10.0,

         'num_leaves': 13,

         'num_threads': 8,

         'tree_learner': 'serial',

         'objective': 'binary',

         'verbosity': 1}



features = [c for c in train_df if c not in ['ID_code', 'target']]

folds = StratifiedKFold(n_splits=2, shuffle=False, random_state=5555555)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

foldz=2

coll = 'sorting by' #len(features)

feature_importance_df = pd.DataFrame()

for f in ftz:

    print(f)

    train_df = pd.read_csv('../input/train.csv')

    #train_df=train_df[cl]

    test_df = pd.read_csv('../input/test.csv')

    #test_df=test_df[cl]

    train_df=train_df.sort_values(by=f).reset_index(drop=True)

    test_df=test_df.sort_values(by=f).reset_index(drop=True)

    target = train_df['target'].reset_index(drop=True)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

        print("Fold :{}".format(fold_ + 1))

        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

        clf = lgb.train(param, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)

        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = features

        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

        fold_importance_df["fold"] = fold_ + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits



    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

    oof_scr = roc_auc_score(target, oof)

    sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

    sub["target"] = predictions

    sub.to_csv(f'{oof_scr}_{foldz}_{coll}_{f}.csv', index=False)