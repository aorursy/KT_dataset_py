import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
import gc

import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

# I don't like SettingWithCopyWarnings ...
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
%matplotlib inline
import os
df_train = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/train.csv')
df_test  = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/test.csv')
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
categorical_features = df_all.select_dtypes(include = ["object"]).columns

frequency_encoding_all = df_all.copy()
del df_all

def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

for col in categorical_features:
    frequency_encoding_all = frequency_encoding(frequency_encoding_all, col)
    
frequency_encoding_all = frequency_encoding_all.drop(categorical_features ,axis=1, inplace=False)
frequency_encoding_train = frequency_encoding_all[:len_train]
frequency_encoding_test = frequency_encoding_all[len_train:]

del frequency_encoding_all;

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = frequency_encoding_train['SalePrice']
del frequency_encoding_train['SalePrice']

if 'SalePrice' in frequency_encoding_test.columns:
    del frequency_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = frequency_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=frequency_encoding_train, n_splits=5)

train_features = [_f for _f in frequency_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(frequency_encoding_train.shape[0])
sub_reg_preds = np.zeros(frequency_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = frequency_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = frequency_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=5000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(frequency_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
frequency_encoding_train['SalePrice'] = y_reg

frequency_encoding_train.to_csv('Fre_mean_encoding_train.csv', index=False)
frequency_encoding_test.to_csv('Fre_mean_encoding_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("Frequency_CV_0.127036.csv", index=False) # submission
df_train = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/train.csv')
df_test  = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/test.csv')
from sklearn.preprocessing import OneHotEncoder
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
categorical_features = df_all.select_dtypes(include = ["object"]).columns

one_hot_encoding = df_all.copy()
one_hot_encoding = pd.get_dummies(one_hot_encoding)
del df_all

one_hot_encoding_train = one_hot_encoding[:len_train]
one_hot_encoding_test = one_hot_encoding[len_train:]
del one_hot_encoding;

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = one_hot_encoding_train['SalePrice']
del one_hot_encoding_train['SalePrice']

if 'SalePrice' in one_hot_encoding_test.columns:
    del one_hot_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = one_hot_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=one_hot_encoding_train, n_splits=5)

train_features = [_f for _f in one_hot_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(one_hot_encoding_train.shape[0])
sub_reg_preds = np.zeros(one_hot_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = one_hot_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = one_hot_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=5000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(one_hot_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("one_hot_encoding_cv_0.127957.csv", index=False) # submission
df_train = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/train.csv')
df_test  = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/test.csv')
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
categorical_features = df_all.select_dtypes(include = ["object"]).columns

label_encoding = df_all.copy()
del df_all
for i in categorical_features:
    label_encoding[i], indexer = pd.factorize(label_encoding[i])
label_encoding_train = label_encoding[:len_train]
label_encoding_test = label_encoding[len_train:]
del label_encoding

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = label_encoding_train['SalePrice']
del label_encoding_train['SalePrice']

if 'SalePrice' in label_encoding_test.columns:
    del label_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = label_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=label_encoding_train, n_splits=5)

train_features = [_f for _f in label_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(label_encoding_train.shape[0])
sub_reg_preds = np.zeros(label_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = label_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = label_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=5000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(label_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("label_encoding_cv_0.128804.csv", index=False) # submission
from sklearn.model_selection import KFold

def mean_k_fold_encoding(col, alpha):
    target_name = 'SalePrice'
    target_mean_global = df_train[target_name].mean()
    
    nrows_cat = df_train.groupby(col)[target_name].count()
    target_means_cats = df_train.groupby(col)[target_name].mean()
    target_means_cats_adj = (target_means_cats*nrows_cat + 
                             target_mean_global*alpha)/(nrows_cat+alpha)
    # Mapping means to test data
    encoded_col_test = df_test[col].map(target_means_cats_adj)
    #임의로 추가 한 부분
    encoded_col_test.fillna(target_mean_global, inplace=True)
    encoded_col_test.sort_index(inplace=True)

    kfold = KFold(n_splits=5, shuffle=True, random_state=1989)
    parts = []
    for trn_inx, val_idx in kfold.split(df_train):
        df_for_estimation, df_estimated = df_train.iloc[trn_inx], df_train.iloc[val_idx]
        nrows_cat = df_for_estimation.groupby(col)[target_name].count()
        target_means_cats = df_for_estimation.groupby(col)[target_name].mean()

        target_means_cats_adj = (target_means_cats * nrows_cat + 
                                target_mean_global * alpha) / (nrows_cat + alpha)

        encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
        parts.append(encoded_col_train_part)
        
    encoded_col_train = pd.concat(parts, axis=0)
    encoded_col_train.fillna(target_mean_global, inplace=True)
    encoded_col_train.sort_index(inplace=True)
    
    return encoded_col_train, encoded_col_test
df_train = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/train.csv')
df_test  = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/test.csv')
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
categorical_features = df_all.select_dtypes(include = ["object"]).columns

mean_encoding = df_all.copy()
del df_all

mean_encoding_train = mean_encoding[:len_train]
mean_encoding_test = mean_encoding[len_train:]
del mean_encoding

#del df_all; gc.collect()
for col in categorical_features:
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    mean_encoding_train[new_feat_name] = temp_encoded_tr.values
    mean_encoding_test[new_feat_name] = temp_encoded_te.values
    
mean_encoding_train = mean_encoding_train.drop(categorical_features, axis=1, inplace=False)
mean_encoding_test = mean_encoding_test.drop(categorical_features, axis=1, inplace=False)
mean_encoding_train.head()
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = mean_encoding_train['SalePrice']
del mean_encoding_train['SalePrice']

if 'SalePrice' in mean_encoding_test.columns:
    del mean_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = mean_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=mean_encoding_train, n_splits=5)

train_features = [_f for _f in mean_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(mean_encoding_train.shape[0])
sub_reg_preds = np.zeros(mean_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = mean_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = mean_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=5000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(mean_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("mean_encoding_cv_0.127619.csv", index=False) # submission
df_train = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/train.csv')
df_test  = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/test.csv')
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
categorical_features = df_all.select_dtypes(include = ["object"]).columns

frequency_encoding_all = df_all.copy()
del df_all

def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

for col in categorical_features:
    frequency_encoding_all = frequency_encoding(frequency_encoding_all, col)
    

mean_encoding_train = frequency_encoding_all[:len_train]
mean_encoding_test = frequency_encoding_all[len_train:]
del frequency_encoding_all

#del df_all; gc.collect()
for col in categorical_features:
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    mean_encoding_train[new_feat_name] = temp_encoded_tr.values
    mean_encoding_test[new_feat_name] = temp_encoded_te.values
    
mean_encoding_train = mean_encoding_train.drop(categorical_features, axis=1, inplace=False)
mean_encoding_test = mean_encoding_test.drop(categorical_features, axis=1, inplace=False)
mean_encoding_train.head()
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = mean_encoding_train['SalePrice']
del mean_encoding_train['SalePrice']

if 'SalePrice' in mean_encoding_test.columns:
    del mean_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = mean_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=mean_encoding_train, n_splits=5)

train_features = [_f for _f in mean_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(mean_encoding_train.shape[0])
sub_reg_preds = np.zeros(mean_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = mean_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = mean_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=5000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(mean_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
mean_encoding_train['SalePrice'] = y_reg

mean_encoding_train.to_csv('Fre_mean_encoding_train.csv', index=False)
mean_encoding_test.to_csv('Fre_mean_encoding_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("Fre_mean_encoding_cv_0.127087.csv", index=False) # submission
df_train = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/train.csv')
df_test  = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/test.csv')
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
categorical_features = df_all.select_dtypes(include = ["object"]).columns

frequency_encoding_all = df_all.copy()
del df_all

def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

for col in categorical_features:
    frequency_encoding_all = frequency_encoding(frequency_encoding_all, col)
    
frequency_encoding_all = frequency_encoding_all.drop(categorical_features ,axis=1, inplace=False)
frequency_encoding_train = frequency_encoding_all[:len_train]
frequency_encoding_test = frequency_encoding_all[len_train:]

del frequency_encoding_all;

def get_folds(df=None, n_splits=10):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = frequency_encoding_train['SalePrice']
del frequency_encoding_train['SalePrice']

if 'SalePrice' in frequency_encoding_test.columns:
    del frequency_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = frequency_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=frequency_encoding_train, n_splits=10)

train_features = [_f for _f in frequency_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(frequency_encoding_train.shape[0])
sub_reg_preds = np.zeros(frequency_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = frequency_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = frequency_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=5000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(frequency_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
frequency_encoding_train['SalePrice'] = y_reg

frequency_encoding_train.to_csv('Fre_mean_encoding_train_10.csv', index=False)
frequency_encoding_test.to_csv('Fre_mean_encoding_test_10.csv', index=False)

test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("Frequency_CV_0.122660_10.csv", index=False) # submission
df_train = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/train.csv')
df_test  = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/test.csv')
from sklearn.preprocessing import OneHotEncoder
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
categorical_features = df_all.select_dtypes(include = ["object"]).columns

one_hot_encoding = df_all.copy()
one_hot_encoding = pd.get_dummies(one_hot_encoding)
del df_all

one_hot_encoding_train = one_hot_encoding[:len_train]
one_hot_encoding_test = one_hot_encoding[len_train:]
del one_hot_encoding;

def get_folds(df=None, n_splits=10):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = one_hot_encoding_train['SalePrice']
del one_hot_encoding_train['SalePrice']

if 'SalePrice' in one_hot_encoding_test.columns:
    del one_hot_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = one_hot_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=one_hot_encoding_train, n_splits=10)

train_features = [_f for _f in one_hot_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(one_hot_encoding_train.shape[0])
sub_reg_preds = np.zeros(one_hot_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = one_hot_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = one_hot_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=5000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(one_hot_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("one_hot_encoding_cv_0.123044_10.csv", index=False) # submission
df_train = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/train.csv')
df_test  = pd.read_csv('../input/week1-exploratory-data-analysis-with-pyhton/test.csv')
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
categorical_features = df_all.select_dtypes(include = ["object"]).columns

label_encoding = df_all.copy()
del df_all
for i in categorical_features:
    label_encoding[i], indexer = pd.factorize(label_encoding[i])
    
label_encoding_train = label_encoding[:len_train]
label_encoding_test = label_encoding[len_train:]
del label_encoding

def get_folds(df=None, n_splits=10):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = label_encoding_train['SalePrice']
del label_encoding_train['SalePrice']

if 'SalePrice' in label_encoding_test.columns:
    del label_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = label_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=label_encoding_train, n_splits=10)

train_features = [_f for _f in label_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(label_encoding_train.shape[0])
sub_reg_preds = np.zeros(label_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = label_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = label_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=5000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(label_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
label_encoding_train['SalePrice'] = y_reg

label_encoding_train.to_csv('label_encoding_train_10.csv', index=False)
label_encoding_test.to_csv('label_encoding_test_10.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("label_encoding_cv_0.125129_10.csv", index=False) # submission