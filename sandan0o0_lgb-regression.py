import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

np.random.seed(2019)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_train_columns = [c for c in df_train.columns if c not in ['ID_code','target']]

target = df_train['target']
param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.05,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 2019}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

oof_lgb_rmse = np.zeros(len(df_train))

predictions_lgb_rmse = np.zeros(len(df_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['target'].values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 20000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=300, early_stopping_rounds=200)

    oof_lgb_rmse[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)



    

    predictions_lgb_rmse += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

print("LGB repeat CV score: {:<8.5f}".format(roc_auc_score(target,oof_lgb_rmse)))
sub_train = pd.DataFrame({"ID_code":df_train["ID_code"].values})

sub_train["target"] = oof_lgb_rmse

sub_train.to_csv("lgb_reg_train.csv", index=False)



sub_test = pd.DataFrame({"ID_code":df_test["ID_code"].values})

sub_test["target"] = predictions_lgb_rmse

sub_test.to_csv("lgb_reg_test.csv", index=False)
def change_neg(df):

    target = df['target']

    if target < 0:

        return 0

    else:

        return target
sub_train['target'] = sub_train.apply(change_neg,axis=1)

sub_test['target'] = sub_test.apply(change_neg,axis=1)
print("CV score: {:<8.5f}".format(roc_auc_score(target,sub_train.target.values)))
sub_train.to_csv("change_neg_train.csv", index=False)

sub_test.to_csv("change_neg_test.csv", index=False)