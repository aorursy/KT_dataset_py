import math, sys, functools, os, codecs, gc, time
import importlib
from pathlib import Path
import numpy as np
import numpy.random as rd
import pandas as pd
from datetime import  datetime as dt
from collections import Counter
import traceback

import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.decomposition import LatentDirichletAllocation


def current_time():
    return dt.strftime(dt.now(),'%Y-%m-%d %H:%M:%S')


def pred(training, testing, y, lgbm_params, submit_file_name):
    # Init predictions
    oof_preds = np.zeros(training.shape[0])
    sub_preds = np.zeros(testing.shape[0])

    # Run KFold
    clf_list = []
    auc_list = []
    feature_importance_df = pd.DataFrame()

    folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED)
    fold_iter = folds.split(training)

    for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):
        print(f"============ start {n_fold+1} fold training ============")

        X_train = training.iloc[trn_idx]
        X_valid = training.iloc[val_idx]
        y_train = np.log1p(y.iloc[trn_idx])
        y_valid = np.log1p(y.iloc[val_idx])
        
        # Lgbm Dataset
        lgtrain = lgb.Dataset(X_train, y_train,
                          feature_name=training.columns.tolist())
        lgvalid = lgb.Dataset(X_valid, y_valid,
                          feature_name=training.columns.tolist())
        # Start fitting
        gbdt_reg = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=20000,
            early_stopping_rounds=100,
            verbose_eval=100,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train', 'valid']
        )

        # evaluation
        preds_train = gbdt_reg.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
        print("RMSE train cv{}:".format(n_fold+1), rmse_train)

        preds_valid = gbdt_reg.predict(X_valid)
        rmse_valid = np.sqrt(mean_squared_error(y_valid, preds_valid))
        print("RMSE valid cv{}:".format(n_fold+1), rmse_valid)
        oof_preds[val_idx] = np.expm1(preds_valid)

        preds_test = gbdt_reg.predict(testing)
        sub_preds += np.expm1(preds_test) / FOLD_NUM

        del lgtrain
        del lgvalid
        gc.collect()
    print("=+"*30)
    rmse_valid = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(oof_preds)))
    print("RMSE valid FULL cv{}:".format(n_fold+1), rmse_valid)
    
    submit = pd.DataFrame(preds_test, columns=["target"], index=testdex)
    submit.to_csv(submit_file_name, index=True, header=True)
    
    return rmse_valid
# Parameter
DATA_PATH = Path('../input/')
SEED = 71
TOPIC_COMP = 20 # number of topics
FOLD_NUM = 5
#===================
# Data Loading
print("data loading")
nrows = None #100 #None
# parse_dates=["activation_date"],
training = pd.read_csv(str(DATA_PATH/'train.csv'), index_col="ID",  nrows=nrows)
traindex = training.index
testing = pd.read_csv(str(DATA_PATH/'test.csv'), index_col="ID",  nrows=nrows)
testdex = testing.index
print("loading finished.")
# target
y = training.target.copy()
del training['target']
y_log = np.log1p(y)
# removing duplicated columns, highly correlated columns
duplicate_cols = ["d60ddde1b", "912836770", "acc5b709d", "f8d75792f", "f333a5f60"]
high_corr_cols = [
"04e06920e", "e90ed19da", "7d72d6787", "4c256f2f9", "871617f50",
"4a3248e89", "15bba6b9e", "3c29aec1e", "4647da55a", "083640132",
"c4ed18259", "8966b4eee", "45713ba5f", "9a3f53be7", "1d0affea2",
"2306bf286", "62d2a813b", "acd155589", "5d26f4d92", "28b21c1d2",
"6dcac05e7", "bfde2aa61", "34d3974de", "598ae7ea9", "e851264a5",
"5619c1297", "0c5eaf8a7", "bacadce94", "22b3e64c8", "224a28832",
"07cfb1624", "8c1e20670", "49131c9e6", "1de1fda2c", "a04f3e320",
"dcc181073", "2e648ce4b", "3c556d78f", "869a169f9", "99258443a"]

print("training.shape", training.shape)
print("testing.shape", testing.shape)

cols = [c for c in training.columns if c not in duplicate_cols + high_corr_cols]
training = training[cols]
testing = testing[cols]

print("training.shape", training.shape)
print("testing.shape", testing.shape)

############################################################################
print("data preprocessing")

# remove constant cols
print("training.shape", training.shape)
print("testing.shape", testing.shape)
nuniq = training.nunique()
constant_col = nuniq.index[nuniq==1].values
training.drop(constant_col, axis=1, inplace=True)
testing.drop(constant_col, axis=1, inplace=True)
print("removed constant columns: {}".format(len(constant_col)))

print("training.shape", training.shape)
print("testing.shape", testing.shape)
df_cols = training.columns.tolist()
print("data preprocessing finished")
lgbm_params = { 
        'objective': 'regression',
        'num_leaves': 60,
        'subsample': 0.61,
        'colsample_bytree': 0.64,
        'min_split_gain': 0.00259,
        'reg_alpha': 0.00514,
        'reg_lambda': 57.148,
        'min_child_weight': 0.7117,
        'verbose': -1,
        'seed': 3,
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'learning_rate': 0.05,
        'metric': 'rmse',
    }
valid_full_rmse = pred(training, testing, y, lgbm_params, "submit_plane_pred.csv")
print(f"RMSLE with only plane features: {valid_full_rmse}")
# Data concatenation
print("Data concatenation.")
df = pd.concat([training, testing], axis=0)

# Converting values to ratio of max value of each cols
print("convert df to percentage value")
df_max = df.max(axis=0)
df_ratio = pd.DataFrame(np.divide(df.values, df_max[np.newaxis, :]))
df_ratio.index = df.index
df_ratio.columns = df.columns

# topic model features
print("start topic modeling")
# assuming occurence count of a word(columns) for each column valuees. (i.e. "5%"" means 5 times occuring on a document)
# 各項目maxに対する比率を出現回数とみなしてLDAの対象データを算出(ex: maxに対し5%の値は5回出現したとみなす)
df_ratio_100 = (df_ratio.fillna(0)*100).astype(int)

# Run LDA
print(current_time(), 'Run LDA')
lda = LatentDirichletAllocation(n_components=TOPIC_COMP, max_iter=10, learning_method='online',
                                learning_offset=50.,random_state=SEED).fit(df_ratio_100)
topic_result = lda.transform(df_ratio_100)

df_topic_result = pd.DataFrame(topic_result, columns=["{0}_{1:02d}".format('tp', i) for i in range(TOPIC_COMP)])
df_topic_result.index = df_ratio_100.index
print(current_time(), "finished topic modeling")

df = df.join(df_topic_result, on="ID", how='left')

# split train and test dataset
training = df.loc[traindex]
testing = df.loc[testdex]
valid_full_rmse_with_tp = pred(training, testing, y, lgbm_params, "submit_topic_pred.csv")
print(f"RMSLE with topic features: {valid_full_rmse_with_tp}")
print(f"ratio of RMSLE with topic feature vs plane{valid_full_rmse_with_tp/valid_full_rmse}")



