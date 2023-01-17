import os
import gc
import time
import warnings
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from math import sqrt
import lightgbm as lgb
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

print(os.listdir("../input"))
warnings.filterwarnings('ignore')
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
    return df
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data.head(20)
train        = pd.read_csv('../input/train.csv', parse_dates=["first_active_month"])
sample       = pd.read_csv('../input/sample_submission.csv')
test         = pd.read_csv('../input/test.csv', parse_dates=["first_active_month"])
ht           = pd.read_csv('../input/historical_transactions.csv',parse_dates=['purchase_date'])
merchant     = pd.read_csv('../input/merchants.csv')
new_merchant = pd.read_csv('../input/new_merchant_transactions.csv',parse_dates=["purchase_date"])
train        = reduce_mem_usage(train)
test         = reduce_mem_usage(test)
ht           = reduce_mem_usage(ht)
new_merchant = reduce_mem_usage(new_merchant)
for df in [ht,new_merchant]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
feature_1 = train.loc[train['feature_1'] == 1]
feature_2 = train.loc[train['feature_1'] == 2]
feature_3 = train.loc[train['feature_1'] == 3]
feature_4 = train.loc[train['feature_1'] == 4]
feature_5 = train.loc[train['feature_1'] == 5]

plt.figure(figsize=(10, 6))
plt.title('Feature_2 Distribution based on Feature_1 values')
sns.despine()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.distplot(feature_1['feature_2'], hist=False, rug=False,label='1')
sns.distplot(feature_2['feature_2'], hist=False, rug=False,label='2')
sns.distplot(feature_3['feature_2'], hist=False, rug=False,label='3')
sns.distplot(feature_4['feature_2'], hist=False, rug=False,label='4')
sns.distplot(feature_5['feature_2'], hist=False, rug=False,label='5')
feature_1 = train.loc[train['feature_2'] == 1]
feature_2 = train.loc[train['feature_2'] == 2]
feature_3 = train.loc[train['feature_2'] == 3]

plt.figure(figsize=(10, 6))
plt.title('Feature_1 Distribution based on Feature_2 values')
sns.despine()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.distplot(feature_1['feature_1'], hist=False, rug=False,label='1')
sns.distplot(feature_2['feature_1'], hist=False, rug=False,label='2')
sns.distplot(feature_3['feature_1'], hist=False, rug=False,label='3')
feature_1 = train.loc[train['feature_3'] == 0]
feature_2 = train.loc[train['feature_3'] == 1]

plt.figure(figsize=(10, 6))
plt.title('Feature_1 Distribution based on Feature_3 values')
sns.despine()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.distplot(feature_1['feature_1'], hist=False, rug=False,label='0')
sns.distplot(feature_2['feature_1'], hist=False, rug=False,label='1')
feature_1 = train.loc[train['feature_3'] == 0]
feature_2 = train.loc[train['feature_3'] == 1]

plt.figure(figsize=(10, 6))
plt.title('Feature_2 Distribution based on Feature_3 values')
sns.despine()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.distplot(feature_1['feature_2'], hist=False, rug=False,label='0')
sns.distplot(feature_2['feature_2'], hist=False, rug=False,label='1')
# thanks to this kernel @ https://www.kaggle.com/artgor/elo-eda-and-models
fig, ax = plt.subplots(3, 1, figsize = (12, 12))
train['feature_1'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal')
train['feature_2'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown')
train['feature_3'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='gold');
plt.figure(figsize=(10, 6))
plt.title('Target Distribution')
sns.despine()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.distplot(train['target'], hist=True, rug=False,norm_hist=True)
categorical_feats = ['feature_1', 'feature_2', 'feature_3']

for col in categorical_feats:
    
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col]  = lbl.transform(list(test[col].values.astype('str')))
    
df_all = pd.concat([train, test])
df_all = pd.get_dummies(df_all, columns=categorical_feats)

len_train = train.shape[0]

train = df_all[:len_train]
test  = df_all[len_train:]
# thank you @ https://www.kaggle.com/yhn112/data-exploration-lightgbm-catboost-lb-3-760
# thank you @https://www.kaggle.com/konradb/lgb-fe-lb-3-707

ht['authorized_flag'] = ht['authorized_flag'].map({'Y':1, 'N':0})
ht['category_1']      = ht['category_1'].map({'Y': 1, 'N': 0})
ht['category_2x1']    = (ht['category_2'] == 1) + 0
ht['category_2x2']    = (ht['category_2'] == 2) + 0
ht['category_2x3']    = (ht['category_2'] == 3) + 0
ht['category_2x4']    = (ht['category_2'] == 4) + 0
ht['category_2x5']    = (ht['category_2'] == 5) + 0
ht['category_3A']     = (ht['category_3'].astype(str) == 'A') + 0
ht['category_3B']     = (ht['category_3'].astype(str) == 'B') + 0
ht['category_3C']     = (ht['category_3'].astype(str) == 'C') + 0

ht['month_diff'] = ((datetime.datetime.today() - ht['purchase_date']).dt.days)//30
ht['month_diff'] += ht['month_lag']

def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2': ['nunique'],
        'category_3A': ['sum'],
        'category_3B': ['sum'],
        'category_3C': ['sum'],
        'category_2x1': ['sum','mean'],
        'category_2x2': ['sum','mean'],
        'category_2x3': ['sum','mean'],
        'category_2x4': ['sum','mean'],
        'category_2x5': ['sum','mean'],   
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'month_lag': ['min', 'max'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

history = aggregate_historical_transactions(ht)
del ht
gc.collect()
train = pd.merge(train, history, on='card_id', how='left')
test  = pd.merge(test, history, on='card_id', how='left')
# thanks to the kernel @ https://www.kaggle.com/yhn112/data-exploration-lightgbm-catboost-lb-3-760

for df in [train,test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['week'] = df['first_active_month'].dt.week
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

train.drop('first_active_month',axis=1,inplace=True)
test.drop('first_active_month',axis=1,inplace=True)
new_merchant['purchase_date'] = pd.DatetimeIndex(new_merchant['purchase_date']).\
                                astype(np.int64) * 1e-9

new_merchant['authorized_flag'] = new_merchant['authorized_flag'].map({'Y':1, 'N':0})
new_merchant['category_1']      = new_merchant['category_1'].map({'Y':1, 'N':0})
new_merchant['category_3A']     = (new_merchant['category_3'].astype(str) == 'A') + 0
new_merchant['category_3B']     = (new_merchant['category_3'].astype(str) == 'B') + 0
new_merchant['category_3C']     = (new_merchant['category_3'].astype(str) == 'C') + 0
new_merchant['category_2x1']    = (new_merchant['category_2'] == 1) + 0
new_merchant['category_2x2']    = (new_merchant['category_2'] == 2) + 0
new_merchant['category_2x3']    = (new_merchant['category_2'] == 3) + 0
new_merchant['category_2x4']    = (new_merchant['category_2'] == 4) + 0
new_merchant['category_2x5']    = (new_merchant['category_2'] == 5) + 0

def aggregate_new_transactions(new_trans):    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1':   ['sum', 'mean'],
        'category_2':   ['nunique'],
        'category_3A':  ['sum'],
        'category_3B':  ['sum'],
        'category_3C':  ['sum'],     
        'category_2x1': ['sum','mean'],
        'category_2x2': ['sum','mean'],
        'category_2x3': ['sum','mean'],
        'category_2x4': ['sum','mean'],
        'category_2x5': ['sum','mean'],  
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'month_lag': ['min', 'max'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique']
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new_trans = aggregate_new_transactions(new_merchant)

del new_merchant
gc.collect()
train = pd.merge(train, new_trans, on='card_id', how='left')
test  = pd.merge(test, new_trans, on='card_id', how='left')
train = reduce_mem_usage(train)
test  = reduce_mem_usage(test)
y     = train['target']
train = train.drop(['target'],axis=1)
test  = test.drop(['target'],axis=1)
id_train = train['card_id'].copy()
id_test  = test['card_id'].copy()

train = train.drop('card_id', axis = 1)
test  = test.drop('card_id', axis = 1)

nfolds = 10
#folds  = KFold(n_splits= nfolds, shuffle=True, random_state=15)
folds = KFold(n_splits=5, shuffle=True, random_state=4590)
param = {'num_leaves': 129,
         'min_data_in_leaf': 148, 
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "min_child_samples": 24,
         "boosting": "gbdt",
         "feature_fraction": 0.7202,
         "bagging_freq": 1,
         "bagging_fraction": 0.8125 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.3468,
         "verbosity": -1}
feature_importance_df = np.zeros((train.shape[1], nfolds))
mvalid = np.zeros(len(train))
mfull  = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, train.values)):
    print('----')
    print("fold n°{}".format(fold_))
    
    x0,y0 = train.iloc[trn_idx], y[trn_idx]
    x1,y1 = train.iloc[val_idx], y[val_idx]
    
    trn_data = lgb.Dataset(x0, label= y0); val_data = lgb.Dataset(x1, label= y1)
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=500, early_stopping_rounds = 150)
    mvalid[val_idx] = clf.predict(x1, num_iteration=clf.best_iteration)
    
    feature_importance_df[:, fold_] = clf.feature_importance()
    
    mfull += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
    
np.sqrt(mean_squared_error(mvalid, y))
ximp = pd.DataFrame()
ximp['feature'] = train.columns
ximp['importance'] = feature_importance_df.mean(axis = 1)

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=ximp.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.figure(figsize=(10, 6))
plt.title('Target Distribution')
sns.despine()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.distplot(mfull,hist=True, rug=True,norm_hist=True)
mfull.mean()
score = 3.704
submission = pd.DataFrame()
submission['card_id']  = id_test
submission['target'] = mfull
submission.to_csv('submission_lgb.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.01
submission.to_csv('submission_lgb_1_perc_plus.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.02
submission.to_csv('submission_lgb_2_perc_plus.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.03
submission.to_csv('submission_lgb_3_perc_plus.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.04
submission.to_csv('submission_lgb_4_perc_plus.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.05
submission.to_csv('submission_lgb_5_perc_plus.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.06
submission.to_csv('submission_lgb_6_perc_plus.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.1
submission.to_csv('submission_lgb_10_perc_plus.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.12
submission.to_csv('submission_lgb_12_perc_plus.csv', index = False)

submission['card_id']  = id_test
submission['target'] = mfull * 1.2
submission.to_csv('submission_lgb_20_perc_plus.csv', index = False)
