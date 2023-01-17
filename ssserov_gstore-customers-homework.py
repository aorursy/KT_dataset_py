# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import json
import lightgbm as lgb
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import copy
import datetime

import sklearn
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

%matplotlib inline
np.random.seed(127)
def load_data(filename, nrows=None, raw=False):
    if raw == True:
        json_columns = ["device", "geoNetwork", "totals", "trafficSource"]

        df = pd.read_csv("../input/gstore-homework/" + filename,
                        converters={column: json.loads for column in json_columns},
                        nrows=nrows)
        
    else:
        df = pd.read_csv("../input/gstore-homework/" + filename, nrows=nrows, index_col=0,
                         dtype={'fullVisitorId': 'str'}, parse_dates=["date"])
        
    return df
X_train = load_data(filename="train_full_clear_corrected.csv", nrows=None, raw=False)
y_train = copy.deepcopy(X_train["transactionRevenue"].values)
y_train_user = X_train[['fullVisitorId', 'transactionRevenue']].groupby('fullVisitorId').sum() # !!!!!
X_train.drop(columns=["transactionRevenue"], inplace=True)

X_test = load_data(filename="test_full_clear_corrected.csv", nrows=None, raw=False)

bounces_train = pd.read_csv("../input/gstore-homework/bounces_train.csv")
bounces_test = pd.read_csv("../input/gstore-homework/bounces_test.csv")

tmp = pd.DataFrame(X_test["fullVisitorId"]).merge(bounces_test, left_index=True, right_index=True)
bounced_users_mask = (tmp[["fullVisitorId", "bounces"]].groupby("fullVisitorId").bounces.value_counts().unstack(fill_value=0).loc[:, 0] == 0)

bounced_users_idxs = tmp.set_index("fullVisitorId")[bounced_users_mask].reset_index()["fullVisitorId"]
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
cat_cols = ["channelGrouping", "browser", 
            "deviceCategory", "operatingSystem", 
            "city", "continent", 
            "country", "metro",
            "networkDomain", "region", 
            "subContinent", "campaign", "medium", "isMobile", "source"]

cat_cols_without_time = ["channelGrouping", "browser", 
            "deviceCategory", "operatingSystem", 
            "city", "continent", 
            "country", "metro",
            "networkDomain", "region", 
            "subContinent", "campaign", "medium", "isMobile", "source"]

num_cols = ["hits", "pageviews", "visitNumber", "newVisits"] 

num_cols_without_time = ["hits", "pageviews", "visitNumber", "newVisits"]

train_features = cat_cols + num_cols
for col in cat_cols:
    print(col)
    lbl = sklearn.preprocessing.LabelEncoder()
    lbl.fit(list(X_train[col].values.astype('str')) + list(X_test[col].values.astype('str')))
    X_train[col] = lbl.transform(list(X_train[col].values.astype('str')))
    X_test[col] = lbl.transform(list(X_test[col].values.astype('str')))


for col in num_cols:
    X_train[col] = X_train[col].astype(float)
    X_test[col] = X_test[col].astype(float)
params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : 20,
    "min_child_samples" : 100,
    "learning_rate" : 0.1, # 0.05,
    "bagging_fraction" : 0.7, # 0.5,
    "feature_fraction" : 0.5, # 0.7,
    "bagging_frequency" : 5,
    "bagging_seed" : 127 # 2018
}


xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.1,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

cat_params = {
    'learning_rate' :0.1,
    'depth' :10,
    'eval_metric' :'RMSE',
    'od_type' :'Iter',
#     'metric_period ' : 50,
    'od_wait' : 20,
    'random_seed' : 42
    
}

forest_params = {
    "n_estimators": 100,
    "min_samples_split": 100,
    "min_samples_leaf": 100,
    "n_jobs": -1,
    "verbose": 1
}
# Cross-validation!
folds = get_folds(df=X_train, n_splits=5)


importances = pd.DataFrame()
oof_lgb_preds = np.zeros(X_train.shape[0])
oof_xgb_preds = np.zeros(X_train.shape[0])
oof_cat_preds = np.zeros(X_train.shape[0])
oof_forest_preds = np.zeros(X_train.shape[0])
lgb_preds = np.zeros((X_test.shape[0], len(folds)))
xgb_preds = np.zeros((X_test.shape[0], len(folds)))
cat_preds = np.zeros((X_test.shape[0], len(folds)))
forest_preds = np.zeros((X_test.shape[0], len(folds)))

merge_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros((X_test.shape[0], len(folds)))


for fold_, (trn_, val_) in enumerate(folds):
    print("Fold:",fold_)
    
    trn_x, trn_y = X_train[train_features].iloc[trn_], y_train[trn_]
    val_x, val_y = X_train[train_features].iloc[val_], y_train[val_]
    
    
    reg = lgb.LGBMRegressor(**params, n_estimators=1100)
    xgb = XGBRegressor(**xgb_params, n_estimators=1000)
    cat = CatBoostRegressor(**cat_params, iterations=1000)
    forest = RandomForestRegressor(**forest_params)
    
    print("-"* 20 + "LightGBM Training" + "-"* 20)
    reg.fit(trn_x, np.log1p(trn_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,verbose=100,eval_metric='rmse')
    print("-"* 20 + "XGboost Training" + "-"* 20)
    xgb.fit(trn_x, np.log1p(trn_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,eval_metric='rmse',verbose=100)
    print("-"* 20 + "Catboost Training" + "-"* 20)
    cat.fit(trn_x, np.log1p(trn_y), eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,use_best_model=True,verbose=100)
    print("-"* 20 + "Forest Training" + "-"* 20)
    forest.fit(trn_x, np.log1p(trn_y))


    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain_lgb'] = reg.booster_.feature_importance(importance_type='gain')
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    # LightGBM
    oof_lgb_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_lgb_preds[oof_lgb_preds < 0] = 0
    lgb_preds_cur = reg.predict(X_test[train_features], num_iteration=reg.best_iteration_)
    lgb_preds_cur[lgb_preds_cur < 0] = 0
    
    # Xgboost
    oof_xgb_preds[val_] = xgb.predict(val_x)
    oof_xgb_preds[oof_xgb_preds < 0] = 0
    xgb_preds_cur = xgb.predict(X_test[train_features])
    xgb_preds_cur[xgb_preds_cur < 0] = 0
    
    # Catboost
    oof_cat_preds[val_] = cat.predict(val_x)
    oof_cat_preds[oof_cat_preds < 0] = 0
    cat_preds_cur = cat.predict(X_test[train_features])
    cat_preds_cur[cat_preds_cur < 0] = 0

    # Forest
    oof_forest_preds[val_] = forest.predict(val_x)
    oof_forest_preds[oof_forest_preds < 0] = 0
    forest_preds_cur = forest.predict(X_test[train_features])
    forest_preds_cur[forest_preds_cur < 0] = 0
    
    
    # merge all prediction
    merge_preds[val_] = oof_lgb_preds[val_] * 0.6 + oof_xgb_preds[val_] * 0.2 +\
    oof_cat_preds[val_] * 0.1 + oof_forest_preds[val_] * 0.1
    
    
    
    lgb_preds[:, fold_] = np.expm1(lgb_preds_cur)
    xgb_preds[:, fold_] = np.expm1(xgb_preds_cur)
    cat_preds[:, fold_] = np.expm1(cat_preds_cur)
    forest_preds[:, fold_] = np.expm1(forest_preds_cur)
                     
    
    
# Without postprocessing (bounces)
    
oof_lgb_preds_user = pd.DataFrame(X_train["fullVisitorId"]).merge(pd.DataFrame(np.expm1(oof_lgb_preds),
                                            columns=["prediction"]), left_index=True, right_index=True) # Expm1 !!!!!
oof_lgb_preds_user = oof_lgb_preds_user.groupby("fullVisitorId").sum()
comparison = pd.merge(y_train_user, oof_lgb_preds_user,
                      on=["fullVisitorId"], how="inner")[["transactionRevenue", "prediction"]]
comparison = np.log1p(comparison.values)
print("Total crossval LGBM rmse: ", mean_squared_error(comparison[:, 0], comparison[:, 1]) ** .5)


oof_xgb_preds_user = pd.DataFrame(X_train["fullVisitorId"]).merge(pd.DataFrame(np.expm1(oof_xgb_preds),
                                            columns=["prediction"]), left_index=True, right_index=True) # Expm1 !!!!!
oof_xgb_preds_user = oof_xgb_preds_user.groupby("fullVisitorId").sum()
comparison = pd.merge(y_train_user, oof_xgb_preds_user,
                      on=["fullVisitorId"], how="inner")[["transactionRevenue", "prediction"]]
comparison = np.log1p(comparison.values)
print("Total crossval XGB rmse: ", mean_squared_error(comparison[:, 0], comparison[:, 1]) ** .5)
                        
    
oof_cat_preds_user = pd.DataFrame(X_train["fullVisitorId"]).merge(pd.DataFrame(np.expm1(oof_cat_preds),
                                            columns=["prediction"]), left_index=True, right_index=True) # Expm1 !!!!!
oof_cat_preds_user = oof_cat_preds_user.groupby("fullVisitorId").sum()
comparison = pd.merge(y_train_user, oof_cat_preds_user,
                      on=["fullVisitorId"], how="inner")[["transactionRevenue", "prediction"]]
comparison = np.log1p(comparison.values)
print("Total crossval CatBoost rmse: ", mean_squared_error(comparison[:, 0], comparison[:, 1]) ** .5)
                        
    
oof_forest_preds_user = pd.DataFrame(X_train["fullVisitorId"]).merge(pd.DataFrame(np.expm1(oof_forest_preds),
                                            columns=["prediction"]), left_index=True, right_index=True) # Expm1 !!!!!
oof_forest_preds_user = oof_forest_preds_user.groupby("fullVisitorId").sum()
comparison = pd.merge(y_train_user, oof_forest_preds_user,
                      on=["fullVisitorId"], how="inner")[["transactionRevenue", "prediction"]]
comparison = np.log1p(comparison.values)
print("Total crossval RF rmse: ", mean_squared_error(comparison[:, 0], comparison[:, 1]) ** .5)
for num in range(len(folds)):
    X_train['predictions_' + "lgb"] = np.expm1(oof_lgb_preds)
    X_train['predictions_' + "xgb"] = np.expm1(oof_xgb_preds)
    X_train['predictions_' + "cat"] = np.expm1(oof_cat_preds)
    X_train['predictions_' + "forest"] = np.expm1(oof_forest_preds)
    
    X_test['predictions_' + "lgb"] = np.mean(lgb_preds, axis=1)
    X_test['predictions_' + "xgb"] = np.mean(xgb_preds, axis=1)
    X_test['predictions_' + "cat"] = np.mean(cat_preds, axis=1)
    X_test['predictions_' + "forest"] = np.mean(forest_preds, axis=1)
# Aggregate data at User level
trn_data = X_train[num_cols_without_time + ['fullVisitorId']].groupby('fullVisitorId').mean()
trn_data[cat_cols_without_time] = X_train[cat_cols_without_time + ['fullVisitorId']].groupby('fullVisitorId').median()
full_list = []
for model in ["lgb", "xgb", "cat", "forest"]:
    trn_pred_list = X_train[['fullVisitorId', 'predictions_' + model]].groupby('fullVisitorId')\
        .apply(lambda df: list(df["predictions_" + model]))\
        .apply(lambda x: {'pred_' + str(i) + "_" + model: pred for i, pred in enumerate(x) if i <= 43})
    
    full_list.append(trn_pred_list)
for num, model in enumerate(["lgb", "xgb", "cat", "forest"]):
    if num == 0:
        trn_all_predictions = pd.DataFrame(list(full_list[num].values), index=trn_data.index)
    else:
        trn_all_predictions = trn_all_predictions.merge(pd.DataFrame(list(full_list[num].values), index=trn_data.index),
                                                        left_index=True, right_index=True)
del trn_pred_list, full_list
# Create a DataFrame with VisitorId as index

for model in ["lgb", "xgb", "cat", "forest"]:
    a = X_train[['fullVisitorId', 'predictions_' + model]].groupby('fullVisitorId')
    tmp = copy.deepcopy(X_train[['fullVisitorId', 'predictions_' + model]])
    tmp["predictions_" + model] = np.log1p(tmp["predictions_" + model])
    tmp = tmp.groupby("fullVisitorId")

    trn_data['t_mean_' + model] = np.log1p(a.mean())
    trn_data['t_median_' + model] = np.log1p(a.median())
    trn_data['t_sum_log_' + model] = tmp.sum()
    trn_data['t_sum_act_' + model] = np.log1p(a.sum())
    
trn_data['t_nb_sess'] = a.count()
# full_data = trn_data
full_data = pd.concat([trn_data, trn_all_predictions], axis=1).fillna(0)

train_2level_features = trn_all_predictions.columns
del trn_data, trn_all_predictions, X_train

# Aggregate data at User level

sub_data = X_test[num_cols_without_time + ['fullVisitorId']].groupby('fullVisitorId').mean()
sub_data[cat_cols_without_time] = X_test[cat_cols_without_time + ['fullVisitorId']].groupby('fullVisitorId').median()
full_list_test = []
for model in ["lgb", "xgb", "cat", "forest"]:
    test_pred_list = X_test[['fullVisitorId', 'predictions_' + model]].groupby('fullVisitorId')\
        .apply(lambda df: list(df["predictions_" + model]))\
        .apply(lambda x: {'pred_' + str(i) + "_" + model: pred for i, pred in enumerate(x) if i <= 43})
    
    full_list_test.append(test_pred_list)
for num, model in enumerate(["lgb", "xgb", "cat", "forest"]):
    if num == 0:
        test_all_predictions = pd.DataFrame(list(full_list_test[num].values), index=sub_data.index)
    else:
        test_all_predictions = test_all_predictions.merge(pd.DataFrame(list(full_list_test[num].values),
                                                                       index=sub_data.index),
                                                        left_index=True, right_index=True)
        
for f in train_2level_features:
    if f not in test_all_predictions.columns:
        test_all_predictions[f] = np.nan
del test_pred_list, full_list_test
# Create a DataFrame with VisitorId as index

for model in ["lgb", "xgb", "cat", "forest"]:
    b = X_test[['fullVisitorId', 'predictions_' + model]].groupby('fullVisitorId')
    tmp2 = copy.deepcopy(X_test[['fullVisitorId', 'predictions_' + model]])
    tmp2["predictions_" + model] = np.log1p(tmp2["predictions_" + model])
    tmp2 = tmp2.groupby("fullVisitorId")

    sub_data['t_mean_' + model] = np.log1p(b.mean())
    sub_data['t_median_' + model] = np.log1p(b.median())
    sub_data['t_sum_log_' + model] = tmp2.sum()
    sub_data['t_sum_act_' + model] = np.log1p(b.sum())
    
sub_data['t_nb_sess'] = b.count()
sub_full_data = pd.concat([sub_data, test_all_predictions], axis=1).fillna(0)
del sub_data, test_all_predictions
params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : 20,
    "min_child_samples" : 100,
    "learning_rate" : 0.01, # 0.05,
    "bagging_fraction" : 0.7, # 0.5,
    "feature_fraction" : 0.5, # 0.7,
    "bagging_frequency" : 5,
    "bagging_seed" : 127 # 2018
}

xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

cat_params = {
    'learning_rate' :0.01,
    'depth' :10,
    'eval_metric' :'RMSE',
    'od_type' :'Iter',
    'od_wait' : 20,
    'random_seed' : 42
    
}
folds = get_folds(df=full_data["pageviews"].reset_index(), n_splits=2)

oof_reg_preds = np.zeros(full_data.shape[0])
oof_reg_preds1 = np.zeros(full_data.shape[0])
oof_reg_preds2 = np.zeros(full_data.shape[0])
sub_preds = np.zeros(sub_full_data.shape[0])
vis_importances = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds):
    print("Fold:",fold_)
    
    trn_x, trn_y = full_data.iloc[trn_], y_train_user['transactionRevenue'].iloc[trn_]
    val_x, val_y = full_data.iloc[val_], y_train_user['transactionRevenue'].iloc[val_]
    
    reg = lgb.LGBMRegressor(**params, n_estimators=1100)
    xgb = XGBRegressor(**xgb_params, n_estimators=1000)
    cat = CatBoostRegressor(**cat_params, iterations=1000)
    print("-"* 20 + "LightGBM Training" + "-"* 20)
    reg.fit(trn_x, np.log1p(trn_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,verbose=100,eval_metric='rmse')
    print("-"* 20 + "XGboost Training" + "-"* 20)
    xgb.fit(trn_x, np.log1p(trn_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,eval_metric='rmse',verbose=100)
    print("-"* 20 + "Catboost Training" + "-"* 20)
    cat.fit(trn_x, np.log1p(trn_y), eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,use_best_model=True,verbose=100)
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = trn_x.columns
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    vis_importances = pd.concat([vis_importances, imp_df], axis=0, sort=False)
    
    # LightGBM
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    lgb_preds = reg.predict(sub_full_data, num_iteration=reg.best_iteration_)
    lgb_preds[lgb_preds < 0] = 0
    
    
    # Xgboost
    oof_reg_preds1[val_] = xgb.predict(val_x)
    oof_reg_preds1[oof_reg_preds1 < 0] = 0
    xgb_preds = xgb.predict(sub_full_data)
    xgb_preds[xgb_preds < 0] = 0
    
    # catboost
    oof_reg_preds2[val_] = cat.predict(val_x)
    oof_reg_preds2[oof_reg_preds2 < 0] = 0
    cat_preds = cat.predict(sub_full_data)
    cat_preds[cat_preds < 0] = 0
    
    sub_preds += (lgb_preds / len(folds)) * 0.6 + (xgb_preds / len(folds)) * 0.3 + (cat_preds / len(folds)) * 0.1

    
print("LGBM Result ", mean_squared_error(np.log1p(y_train_user['transactionRevenue']), oof_reg_preds) ** .5)
print("XGBoost Result", mean_squared_error(np.log1p(y_train_user['transactionRevenue']), oof_reg_preds1) ** .5)
print("CatBoost Result", mean_squared_error(np.log1p(y_train_user['transactionRevenue']), oof_reg_preds2) ** .5)
# Transaction level!

importances['gain_log'] = np.log1p(importances['gain_lgb'])
mean_gain = importances[['gain_lgb', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain_lgb'])

plt.figure(figsize=(8, 10))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False).iloc[:300])
# User level!

vis_importances['gain_log'] = np.log1p(vis_importances['gain'])
mean_gain = vis_importances[['gain', 'feature']].groupby('feature').mean()
vis_importances['mean_gain'] = vis_importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 10))
sns.barplot(x='gain_log', y='feature', data=vis_importances.sort_values('mean_gain', ascending=False).iloc[:300])
sub_df = pd.DataFrame({"fullVisitorId": sub_full_data.reset_index()["fullVisitorId"].values})
sub_df["PredictedLogRevenue"] = sub_preds
idxs = sub_df["fullVisitorId"].isin(bounced_users_idxs)
sub_df.loc[idxs, "PredictedLogRevenue"] = 0
sub_df.to_csv('two_levels_ensemble.csv', index=False)