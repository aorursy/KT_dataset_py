# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
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

def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)
# This contains training data with features and targe variable. Names are masked. 
data_train = pd.read_csv('../input/train.csv',parse_dates=['first_active_month'])

# This file contains test data 
data_test = pd.read_csv('../input/test.csv',parse_dates=['first_active_month'])

# This file contains additional information about all merchants / merchant_ids in the dataset
#merchants = pd.read_csv('../input/merchants.csv')
new_merch_trans = pd.read_csv('../input/new_merchant_transactions.csv',parse_dates =['purchase_date'])

# This file contains up to 3 months' worth of historical transactions for each card_id
hist_trans = pd.read_csv('../input/historical_transactions.csv',parse_dates =['purchase_date'])
print('Null data in training data')
print(data_train.isnull().sum())
print('Null data in test data')
print(data_test.isnull().sum())
dummydate = data_test['first_active_month'][0]
data_test['first_active_month'].fillna(dummydate,inplace=True)
print (dummydate)
# Check what is inside of these files first. 
data_train.head()
#data_train.describe()
data_test.head()
plt.scatter(range(data_train.shape[0]), np.sort(data_train['target'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.show()
sns.set()
data_train['target'].plot(kind='hist')
plt.xlabel('Target')
plt.title('Variation of Target Values')
# This file contains additional information about all merchants / merchant_ids in the dataset.
# Important features here are category_id, merchant_group_id, subsector_id, city_id, state_id
#merchants.head()
# This file contains up to 3 months' worth of historical transactions for each card_id
# So this builds some history for every card user at set number of merchants. 
hist_trans.head()
# This file contains two months worth of data for each card_id containing ALL purchases that card_id made at merchant_ids 
# that were not visited in the historical data.

# So apart from historical data, new history checks if the card user is spending more on new categories

new_merch_trans.head()
# Map the columns with helping function from 
# Ref: https://www.kaggle.com/fabiendaniel/elo-world
def binarize(df):
    for column in ['authorized_flag', 'category_1']:
        df[column] = df[column].map({'Y':1, 'N':0})
    return df

hist_trans_prep = binarize(hist_trans)
new_merch_trans_prep = binarize(new_merch_trans)
hist_trans_prep = reduce_mem_usage(hist_trans_prep)
new_merch_trans_prep = reduce_mem_usage(new_merch_trans_prep)
def split_time2(df):
    df['pd_year'] = df.purchase_date.dt.year
    df['pd_month'] = df.purchase_date.dt.month
    df['pd_day_of_year'] = df.purchase_date.dt.dayofyear
    df['pd_day_of_week'] = df.purchase_date.dt.dayofweek
    df['pd_hour'] = df.purchase_date.dt.hour
    return df

hist_trans_prep = split_time2(hist_trans_prep)
new_merch_trans_prep = split_time2(new_merch_trans_prep)
# Drop categorical and date columns
hist_trans_prep.drop('purchase_date',1,inplace=True)
new_merch_trans_prep.drop('purchase_date',1,inplace=True)
# Categorical data dummy creation for transaction data

# Create categorical columns
features2 = ['category_2', 'category_3']

# get dummies
hist_trans_prep = pd.get_dummies(hist_trans_prep,columns=features2)
new_merch_trans_prep = pd.get_dummies(new_merch_trans_prep,columns=features2)
hist_trans_prep = reduce_mem_usage(hist_trans_prep)
new_merch_trans_prep = reduce_mem_usage(new_merch_trans_prep)
def split_time(df):
    df['fac_year'] = df.first_active_month.dt.year
    df['fac_month'] = df.first_active_month.dt.month
    df['fac_day_of_year'] = df.first_active_month.dt.dayofyear
    df['fac_day_of_week'] = df.first_active_month.dt.dayofweek
    return df
    
data_train_prep = split_time(data_train)
data_test_prep = split_time(data_test)
data_train_prep.drop('first_active_month',1,inplace=True)
data_test_prep.drop('first_active_month',1,inplace=True)

data_train_prep.head()
#data_train_prep = data_train_prep[data_train_prep['target']>-20]
#final_size = data_train_prep.shape[0]
#original_size= data_train.shape[0]
#print('Percentage data dropped after preparation',(original_size-final_size)/original_size*100)
# Create categorical columns
features = ['feature_1','feature_2','feature_3']

# get dummies
data_train_prep = pd.get_dummies(data_train_prep,columns=features)
data_test_prep = pd.get_dummies(data_test_prep,columns=features)
data_train_prep.head()
data_test_prep.head()
hist_trans_prep.head()
#hist_trans_prep.describe()
new_merch_trans_prep.head()
#print('Null data in historical transaction data')
#print(hist_trans.isnull().sum())
#print('Null data in new data')
#print(new_merch_trans.isnull().sum())
agg_fun = {'authorized_flag': ['sum', 'mean']}
hist_auth_mean = hist_trans_prep.groupby(['card_id']).agg(agg_fun)
hist_auth_mean.head()
hist_auth_mean.columns = ['_'.join(col).strip() for col in hist_auth_mean.columns.values]
hist_auth_mean.reset_index(inplace=True)
# Sum is total of authorized transactions and mean is % of total transcations
hist_auth_mean.head()
# Filtering out authorized transactions only for historical data
hist_trans_prep_auth = hist_trans_prep[hist_trans_prep['authorized_flag']==1]
hist_trans_prep_auth.head()
def aggregate_transactions(history):
    agg_func = {
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min'],
        'installments': ['sum', 'mean', 'max', 'min'],
        'pd_month': ['mean', 'max', 'min'],
        'pd_year': [np.ptp, 'min', 'max'],
        'month_lag': ['min', 'max']
        }
    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    # Get total transaction for each card and then add it as a new column
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history
history_prep_auth_agg = aggregate_transactions(hist_trans_prep_auth)
history_prep_auth_agg.columns = ['hist_' + c if c != 'card_id' else c for c in history_prep_auth_agg.columns]
history_prep_auth_agg[:5]
new_merch_prep_agg = aggregate_transactions(new_merch_trans_prep)
new_merch_prep_agg.columns = ['new_' + c if c != 'card_id' else c for c in new_merch_prep_agg.columns]
new_merch_prep_agg[:5]
data_train_prep_agg = pd.merge(data_train_prep, history_prep_auth_agg, on='card_id', how='left')
data_test_prep_agg = pd.merge(data_test_prep, history_prep_auth_agg, on='card_id', how='left')

data_train_prep_agg = pd.merge(data_train_prep_agg, new_merch_prep_agg, on='card_id', how='left')
data_test_prep_agg = pd.merge(data_test_prep_agg, new_merch_prep_agg, on='card_id', how='left')
data_train_final = pd.merge(data_train_prep_agg, hist_auth_mean, on='card_id', how='left')
data_test_final = pd.merge(data_test_prep_agg, hist_auth_mean, on='card_id', how='left')
data_train_final.head()
data_train_final.insert(1, 'age_months', ((data_train_final['hist_pd_year_max']-data_train_final['fac_year'])*12+ (data_train_final['hist_pd_month_max']-data_train_final['fac_month'])).astype(int))
data_test_final.insert(1, 'age_months', ((data_test_final['hist_pd_year_max']-data_test_final['fac_year'])*12+ (data_test_final['hist_pd_month_max']-data_test_final['fac_month'])).astype(int))
data_train_final.head()
data_test_final.head()
y_train = data_train_final['target']

card_ids_train = data_train_final['card_id']
card_ids_test = data_test_final['card_id']

X_train = data_train_final.drop(['target','card_id'],axis=1)
X_test = data_test_final.drop(['card_id'],axis=1)
#from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
#scaler = MinMaxScaler() # default=(0, 1)
#numerical = ['age_months', 'fac_year', 'fac_month', 'fac_day_of_year', 'fac_day_of_week']
#num_features = [c for c in X_train.columns if not ('feature_' in c or 'ptp' in c)]
#print (num_features)
#print (len(num_features))

#X_train[numerical] = scaler.fit_transform(X_train[numerical])
#X_test[numerical] = scaler.fit_transform(X_test[numerical])
#X_train = reduce_mem_usage(X_train)
#X_test = reduce_mem_usage(X_test)
card_ids_test.head()
#print('Null data in historical transaction data')
#print(X_test.isnull().sum())
#print('Null data in new data')
#print(new_merch_trans.isnull().sum())
# Baseline Model with Linear Regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from math import sqrt

#Reg_model = LinearRegression()
#Reg_model.fit(X_train, y_train)


#y_pred = Reg_model.predict(X_test)
# With linear model score is 3.86
import lightgbm as lgb

param = {'num_leaves': 120,
         'min_data_in_leaf': 148, 
         'objective':'regression',
         "metric" : "rmse",
         'max_depth': 9,
         'learning_rate': 0.005,
         "min_child_samples": 24,
         "boosting": "gbdt",
         "feature_fraction": 0.7202,
         "bagging_freq": 1,
         "bagging_fraction": 0.8125 ,
         "bagging_seed": 11,
         "metric": 'rmse',
        # "lambda_l1": 0.3468,
         "random_state": 4590,
         "verbosity": 1}

param2 = {'num_leaves': 400,
         'min_data_in_leaf': 148, 
         'objective':'regression',
         "metric" : "rmse",
         'max_depth': 9,
         'learning_rate': 0.01,
         "min_child_samples": 24,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 4,
         "random_state": 4590,
         "verbosity": 1}

features = [c for c in X_train.columns if c not in ['card_id']]

# List of categorical features starting with feature_
categ_feats = [c for c in features if 'feature_' in c]
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold

import time


folds = KFold(n_splits=5, shuffle=True, random_state=4520)
#folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4520)

oof = np.zeros(len(X_train))
predictions = np.zeros(len(X_test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(X_train.iloc[trn_idx][features],
                           label=y_train.iloc[trn_idx],
                           categorical_feature=categ_feats)
    val_data = lgb.Dataset(X_train.iloc[val_idx][features],
                           label=y_train.iloc[val_idx],
                           categorical_feature=categ_feats)

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds = 200)
    
    oof[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(X_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, y_train)**0.5))
sub_df = pd.DataFrame({"card_id":card_ids_test.values})
sub_df['target'] = predictions
sub_df.to_csv("submit_lgbm.csv", index=False)
import xgboost as xgb

xgb_params = {'eta': 0.005, 
              'max_depth': 5, 
              'subsample': 0.8, 
              'colsample_bytree': 0.8, 
              'objective': 'reg:linear', 
              'eval_metric': 'rmse',
              'verbosity':0
              }

FOLDs = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
predictions_xgb = np.zeros(len(X_test))
for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(X_train, y_train.values)):
    trn_data = xgb.DMatrix(data=X_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
    val_data = xgb.DMatrix(data=X_train.iloc[val_idx], label=y_train.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_round = 500
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=50, verbose_eval=100)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)
    predictions_xgb += xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_splits
    
print("CV score: {:<8.5f}".format(mean_squared_error(oof_xgb, y_train)**0.5))
sub_df2 = pd.DataFrame({"card_id":card_ids_test.values})
sub_df2['target'] = 0.5*predictions+0.5*predictions_xgb
sub_df2.to_csv("submit_ens.csv", index=False)
results_ens = pd.read_csv('submit_ens.csv')
results_lgbm = pd.read_csv('submit_lgbm.csv')
#results_ens.head()
sns.set()
results_ens['target'].plot(kind='hist',bins=10)
results_lgbm['target'].plot(kind='hist',alpha=0.5,bins=10)

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')