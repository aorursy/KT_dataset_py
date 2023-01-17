# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pyarrow.parquet as pq
import os
print(os.listdir("../input"))

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.metrics import mean_squared_error, matthews_corrcoef

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb

# Any results you write to the current directory are saved as output
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
#     return df
meta_train = pd.read_csv('../input/metadata_train.csv')
# meta_test = pd.read_csv('../input/metadata_test.csv')
reduce_mem_usage(meta_train)
# reduce_mem_usage(meta_test)
meta_train.shape
%%time 
# Start small, load in 500 signals and calculate some basic aggregates (mean, sum)
# Read in 500 signals
# Each column contains one signal
subset_train_0_500 = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(500)]).to_pandas()
# Function to add a few basic aggregations for the different signal ids
def add_cols(df, df_sig):
    for signal in df['signal_id'].tolist():
        df.loc[df['signal_id']==signal,'signal_mean'] = np.mean(df_sig[str(signal)])
        df.loc[df['signal_id']==signal,'signal_sum'] = np.sum(df_sig[str(signal)])
        df.loc[df['signal_id']==signal,'signal_median'] = np.median(df_sig[str(signal)])
        df.loc[df['signal_id']==signal,'signal_ptp'] = np.ptp(df_sig[str(signal)])
meta_train['signal_mean'] = 0
meta_train['signal_sum'] = 0
meta_train['signal_median'] = 0
meta_train['signal_ptp'] = 0
meta_train.head()
meta_train_0_500 = meta_train[:500]
%%time
add_cols(meta_train_0_500,subset_train_0_500)
reduce_mem_usage(meta_train_0_500)
meta_train_0_500.head()
meta_train_0_500.tail()
target = meta_train_0_500['target']
meta_train_0_500.drop('target', axis=1, inplace=True)
features = meta_train_0_500.columns
# def lgb_mcc_score(y_hat, data):
#     y_true = data.get_label()
#     y_hat = np.round(y_hat)
#     return 'mcc', matthews_corrcoef(y_true, y_hat), True

# # def lgb_mcc(preds, dtrain):
# #     THRESHOLD = 0.5
# #     labels = dtrain.get_label()
# #     return 'mcc', matthews_corrcoef(labels, preds >= THRESHOLD)


param = {'num_leaves': 60,
         'min_data_in_leaf': 60, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.1,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "random_state": 42,
         "verbosity": -1}
max_iter=5
gc.collect()
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(meta_train_0_500))
# categorical_columns = [c for c in categorical_columns if c not in ['MachineIdentifier']]
# features = [c for c in train.columns if c not in ['MachineIdentifier']]
# predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()

score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(meta_train_0_500.values, target.values)):
    print("Fold No.{}".format(fold_+1))
    trn_data = lgb.Dataset(meta_train_0_500.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
#                            categorical_feature = categorical_columns
                          )
    val_data = lgb.Dataset(meta_train_0_500.iloc[val_idx][features],
                           label=target.iloc[val_idx],
#                            categorical_feature = categorical_columns
                          )
    evals_result = {}
    
    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds = 200)
#                     feval = lgb_mcc_score,
#                     evals_result = evals_result)
    
    oof[val_idx] = clf.predict(meta_train_0_500.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
#     lgb.plot_metric(evals_result, metric='mcc')

    score[fold_] = metrics.roc_auc_score(target.iloc[val_idx], oof[val_idx])
    if fold_ == max_iter - 1: break
        
if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(metrics.roc_auc_score(target, oof)))
else:
     print("CV score: {:<8.5f}".format(sum(score) / max_iter))
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
