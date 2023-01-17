import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

import os
print(os.listdir("../input"))

import seaborn as sns
sns.set(style='white', context='notebook', palette='Set2')
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
train_df = pd.read_csv("../input/train_V2.csv")
test_df = pd.read_csv("../input/test_V2.csv")
train_df.head()
train_df.describe()
test_df.describe()
pd.DataFrame({'train':train_df.isna().sum(), 'test':test_df.isna().sum()})
train_df = train_df.dropna()
drop_features = ["Id", "groupId", "matchId"]
feats = [f for f in train_df.columns if f not in drop_features]

plt.figure(figsize=(18,16))
sns.heatmap(train_df[feats].corr(), linewidths=0.1,vmax=1.0,
               square=True, linecolor='white', annot=True, cmap="RdBu")
plt.figure(figsize=(12,6))
sns.distplot(train_df['winPlacePerc'].values, bins=100, kde=False)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
dataset = pd.concat([train_df, test_df], sort=True)
dataset = reduce_mem_usage(dataset)
plt.figure(figsize=(12,6))
plt.title('Number of Team Members')
tmp = dataset.groupby(['matchId','groupId'])['Id'].agg('count')
sns.countplot(tmp)
plt.figure(figsize=(12,6))
plt.title('Number of Team Members')
ax = sns.countplot(x='matchType', data=dataset)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
# https://www.kaggle.com/rejasupotaro/effective-feature-engineering
def min_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId','groupId'])[features].min()
    return df.merge(agg, suffixes=['', '_min'], how='left', on=['matchId', 'groupId'])

def max_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].max()
    return df.merge(agg, suffixes=['', '_max'], how='left', on=['matchId', 'groupId'])

def sum_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].sum()
    return df.merge(agg, suffixes=['', '_sum'], how='left', on=['matchId', 'groupId'])

def median_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])

def mean_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])

def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])
dataset = pd.concat([train_df, test_df], sort=True)
dataset = reduce_mem_usage(dataset)

# dataset = mean_by_team(dataset)
dataset = rank_by_team(dataset)
gc.collect()

dataset.head()
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error
def oof_model_preds(df, model, num_folds, params):
    # Divide in training/validation and test data
    train_df = df[df['winPlacePerc'].notnull()]
    test_df = df[df['winPlacePerc'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    drop_features = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']    
    feats = [f for f in train_df.columns if f not in drop_features]

    # Create model
    if num_folds == 1:
        train_x, valid_x, train_y, valid_y = train_test_split(train_df[feats], train_df['winPlacePerc'], test_size=0.2, random_state=1001)
        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric= 'mae', verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'])

        oof_preds = model.predict(train_df[feats])
        sub_preds = model.predict(test_df[feats])

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('MAE : %.6f' % (mean_absolute_error(train_df['winPlacePerc'], oof_preds)))
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    # Cross validation model
    elif num_folds > 1:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['winPlacePerc'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['winPlacePerc'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['winPlacePerc'].iloc[valid_idx]

            model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric= 'mae', verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'])

            oof_preds[valid_idx] = model.predict(valid_x)
            sub_preds += model.predict(test_df[feats]) / folds.n_splits

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = model.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(valid_y, oof_preds[valid_idx])))
            del train_x, train_y, valid_x, valid_y
            gc.collect()

    print('Full MAE score %.6f' % mean_absolute_error(train_df['winPlacePerc'], oof_preds))
    return oof_preds, sub_preds, feature_importance_df
import lightgbm as lgb

params = {
    'num_leaves': 144,
    'learning_rate': 0.1,
    'n_estimators': 800,
    'max_depth':12,
    'max_bin':55,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'feature_fraction':0.9,
    'verbose':50, 
    'early_stopping_rounds':100
    }

# LightGBM parameters
lgbm_reg = lgb.LGBMRegressor(num_leaves=params['num_leaves'], learning_rate=params['learning_rate'], 
                    n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                    max_bin = params['max_bin'], bagging_fraction = params['bagging_fraction'], 
                    bagging_freq = params['bagging_freq'], feature_fraction = params['feature_fraction'],
                   )

lgb_oof_preds, lgb_sub_preds, lgb_feature_importance_df = oof_model_preds(dataset, lgbm_reg, num_folds=4, params=params)
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
display_importances(lgb_feature_importance_df)
sub = pd.DataFrame()
sub['Id'] = test_df['Id']
sub['winPlacePerc'] = lgb_sub_preds
sub['winPlacePerc'][sub['winPlacePerc'] > 1] = 1

sub.to_csv('lgb_submission.csv',index=False)
