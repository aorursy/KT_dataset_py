# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import gc
gc.enable()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/gstore-data-cleaning/starting_point.csv', 
                 parse_dates=['date'],
                 dtype={'fullVisitorId':'str'},
                 infer_datetime_format=True)
df.loc[(df.is_train == 1) & (df.totals_transactionRevenue.isnull()), 'totals_transactionRevenue'] = 0
random_seed = 42
def rmse(actual, preds):
    return mean_squared_error(actual, preds) ** (.5)
def generate_user_level_features(df):
    agg_dict = {'totals_hits':['sum'],
                'totals_pageviews':['sum'],
                'totals_transactionRevenue':lambda x: np.log1p(x.sum())}
    df_agg = df.groupby(['is_train', 'fullVisitorId']).agg(agg_dict)
    df_agg.columns = ['totals_hits', 'totals_pageviews', 'target']
    df_agg.reset_index('is_train', inplace=True)
        
    return df_agg[df_agg.is_train == 1].drop('is_train', axis=1), df_agg[df_agg.is_train == 0].drop(['is_train', 'target'], axis=1)

def generate_visit_level_features(df):
    df['target'] = np.log1p(df['totals_transactionRevenue'])
    return df.set_index('fullVisitorId').loc[lambda x: x.is_train == 1, ['totals_hits', 'totals_pageviews', 'target']], df.set_index('fullVisitorId').loc[lambda x: x.is_train == 0, ['totals_hits', 'totals_pageviews']]


def kfold_crossval(df, target, stratified=True, nfolds=4):
    if stratified:
        kf = StratifiedKFold(nfolds, random_state=random_seed)
    else:
        kf = KFold(nfolds, random_state=random_seed)
    stratified_target = [1 if x >0 else 0 for x in target.values]
    return kf.split(df.values, stratified_target)

def agg_user_level(df, preds, targets=None):
    if targets is not None:
        df['target'] = targets
    df['PredictedLogRevenue'] = preds
    return df

def agg_visit_level(df, preds, targets=None):
    if targets is not None:
        df['target'] = np.expm1(targets)
        cols = ['target', 'PredictedLogRevenue']
    else:
        cols = ['PredictedLogRevenue']
    df['PredictedLogRevenue'] = np.expm1(preds)
    df_tmp = df[cols].reset_index().groupby('fullVisitorId').sum().apply(lambda x: np.log1p(x))
    return df_tmp
    
def lgbm_predict(train_x, train_y, valid_x, valid_y, test_x):
    lgbm_params = {"objective" : "regression", "metric" : "rmse", 
                   "max_depth": 8, "min_child_samples": 20, "reg_alpha": 0.2, "reg_lambda": 0.2,
                   "num_leaves" : 257, "learning_rate" : 0.1, "subsample" : 0.9, "colsample_bytree" : 0.9, 
                   "subsample_freq ": 5, 'n_estimators':5000, 'random_state':random_seed}
    
    clf = LGBMRegressor(**lgbm_params)
    
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'rmse', verbose= 500, early_stopping_rounds= 100)
    
    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = train_x.columns
    fold_importance_df_lgb["importance"] = clf.feature_importances_
    
    val_preds = clf.predict(valid_x, num_iteration=clf.best_iteration_)
    
    test_preds = clf.predict(test_x, num_iteration=clf.best_iteration_)
    
    return np.clip(val_preds, 0, None), np.clip(test_preds, 0, None), fold_importance_df_lgb
    
def cv_predict(df, feature_generator, validation_generator, aggregator, predictor):
    train_df, test_df = feature_generator(df)
    
    split_idx = validation_generator(train_df.drop('target', axis=1), train_df['target'])
    
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    hold = []
    feature_imp = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(split_idx):
        train_x, train_y = train_df.drop('target', axis=1).iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df.drop('target', axis=1).iloc[valid_idx], train_df['target'].iloc[valid_idx]
        
        valid_preds, test_preds, fold_importance_df_lgb = predictor(train_x, train_y, valid_x, valid_y, test_df)
        
        sub_preds += test_preds
        
        agg_valid_x = aggregator(valid_x, valid_preds, valid_y)
        
        print('Fold {} : {}'.format(n_fold, rmse(agg_valid_x['target'].values, agg_valid_x['PredictedLogRevenue'].values)))
        
        fold_importance_df_lgb["fold"] = n_fold
        
        hold.append(agg_valid_x)
        feature_imp.append(fold_importance_df_lgb)
    oof_df = pd.concat(hold)
    df_feature_imp = pd.concat(feature_imp)
    print('Full : {}'.format(rmse(oof_df['target'].values, oof_df['PredictedLogRevenue'].values)))
    
    sub_preds  = sub_preds/(n_fold + 1)
    
    agg_test_X = aggregator(test_df, sub_preds)
    
    cols = df_feature_imp[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index
    best_features_lgb = df_feature_imp.loc[df_feature_imp.feature.isin(cols)]
    plt.figure(figsize=(14,10))
    sns.barplot(x="importance", y="feature", data=best_features_lgb.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
        
    return agg_test_X, oof_df , feature_imp   
        
        
    
    
# #### visit level predicitons
# final_df, oof_df, feature_imp = cv_predict(df, generate_visit_level_features, kfold_crossval, agg_visit_level, lgbm_predict)

# final_df[['PredictedLogRevenue']].to_csv('lgbm_visit_preds.csv')

# oof_df[['target', 'PredictedLogRevenue']].to_csv('oof_lgbm_visit_preds.csv')
#### user level preds
final_df, oof_df, feature_imp = cv_predict(df, generate_user_level_features, kfold_crossval, agg_user_level, lgbm_predict)

final_df[['PredictedLogRevenue']].to_csv('lgbm_user_preds.csv')

oof_df[['target', 'PredictedLogRevenue']].to_csv('oof_lgbm_user_preds.csv')
