%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import pickle
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas();

import glob
def get_path(str, first=True, parent_dir='../input/**/'):
    res_li = glob.glob(parent_dir+str)
    return res_li[0] if first else res_li
%%time
train_df = feather.read_dataframe(get_path('train.ftr'))
test_df = feather.read_dataframe(get_path('test.ftr'))
def proc_column_name(df):
    cols = []
    for c in df.columns:
        if '.' in c:
            cols.append(c.replace('.', '_'))
        else:
            cols.append(c)
    df.columns = cols
    return df

train_df = proc_column_name(train_df)
test_df = proc_column_name(test_df)
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error

DEV = True
DEV = False

PATH = '../input/google-analytics-customer-revenue-prediction/'
NUM_ROUNDS = 20000 if not DEV else 20
VERBOSE_EVAL = 10 if not DEV else 10
STOP_ROUNDS = 100 if not DEV else 10
N_SPLITS = 5 if not DEV else 3

json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
def read_parse_dataframe(file_name):
    #full path for the data file
    path = PATH + file_name
    #read the data file, convert the columns in the list of columns to parse using json loader,
    #convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path, 
        converters={column: json.loads for column in cols_to_parse}, 
        dtype={'fullVisitorId': 'str'})
    #parse the json-type columns
    for col in cols_to_parse:
        #each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        #json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        #we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
    return data_df

def process_date_time(data_df):
    print("process date time ...")
    data_df['date'] = data_df['date'].astype(str)
    data_df["date"] = data_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    data_df["date"] = pd.to_datetime(data_df["date"])   
    data_df["year"] = data_df['date'].dt.year
    data_df["month"] = data_df['date'].dt.month
    data_df["day"] = data_df['date'].dt.day
    data_df["weekday"] = data_df['date'].dt.weekday
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['month_unique_user_count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
    data_df['day_unique_user_count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday_unique_user_count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')
    return data_df

def process_format(data_df):
    print("process format ...")
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        data_df[col] = data_df[col].astype(float)
    data_df['trafficSource_adwordsClickInfo_isVideoAd'].fillna(True, inplace=True)
    data_df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    return data_df
    
def process_device(data_df):
    print("process device ...")
    data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
    data_df['browser_operatingSystem'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
    return data_df

def process_totals(data_df):
    print("process totals ...")
    data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
    data_df['totals_hits'] = np.log1p(data_df['totals_hits'])
    data_df['totals_pageviews'] = np.log1p(data_df['totals_pageviews'].fillna(0))
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
    return data_df

def process_geo_network(data_df):
    print("process geo network ...")
    data_df['sum_pageviews_per_network_domain'] = data_df.groupby(
        'geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    data_df['count_pageviews_per_network_domain'] = data_df.groupby(
        'geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    data_df['mean_pageviews_per_network_domain'] = data_df.groupby(
        'geoNetwork_networkDomain')['totals_pageviews'].transform('mean')
    data_df['sum_hits_per_network_domain'] = data_df.groupby(
        'geoNetwork_networkDomain')['totals_hits'].transform('sum')
    data_df['count_hits_per_network_domain'] = data_df.groupby(
        'geoNetwork_networkDomain')['totals_hits'].transform('count')
    data_df['mean_hits_per_network_domain'] = data_df.groupby(
        'geoNetwork_networkDomain')['totals_hits'].transform('mean')
    return data_df
%%time
#Feature processing
## Load data
# train_df = read_parse_dataframe('train.csv')
train_df = process_date_time(train_df)
# test_df = read_parse_dataframe('test.csv')
test_df = process_date_time(test_df)

## Drop columns
cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)

###only one not null value
train_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)

###converting columns format
train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].astype(float)
train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].fillna(0)
train_df['totals_transactionRevenue'] = np.log1p(train_df['totals_transactionRevenue'])
%%time
## Features engineering
train_df = process_format(train_df)
train_df = process_device(train_df)
train_df = process_totals(train_df)
train_df = process_geo_network(train_df)

test_df = process_format(test_df)
test_df = process_device(test_df)
test_df = process_totals(test_df)
test_df = process_geo_network(test_df)
%%time
## Categorical columns
print("process categorical columns ...")
num_cols = [
    'month_unique_user_count', 'day_unique_user_count', 'weekday_unique_user_count',
    'visitNumber', 'totals_hits', 'totals_pageviews', 
    'mean_hits_per_day', 'sum_hits_per_day', 'min_hits_per_day', 'max_hits_per_day', 'var_hits_per_day',
    'sum_pageviews_per_network_domain', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain',
    'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'mean_hits_per_network_domain'
]
            
not_used_cols = ["visitNumber", "date", "fullVisitorId", "sessionId", 
        "visitId", "visitStartTime", 'totals_transactionRevenue', 'trafficSource_referralPath']
cat_cols = [col for col in train_df.columns if col not in num_cols and col not in not_used_cols]
for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
# Model
print("prepare model ...")
#train_df = train_df.sort_values('date')
train_df = train_df.sort_values('visitStartTime')

fullVisitorId = train_df['fullVisitorId'].values
group = LabelEncoder().fit_transform(fullVisitorId)

X = train_df.drop(not_used_cols, axis=1)
y = train_df['totals_transactionRevenue']
X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)
## customized functions
## https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
## https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py

def fobj(preds, dataset): # -> grad, hess
    labels = dataset.get_label()
    ids = dataset.ids
    preds[preds<0] = 0
    pairs = dict(y_true=np.expm1(labels), y_pred=np.expm1(preds), ids=ids)
    pairs = pd.DataFrame(pairs)
    errors = np.log1p(pairs.groupby('ids').sum()).reset_index()
    errors['error'] = errors['y_pred'] - errors['y_true']
    pairs = pairs.merge(errors[['ids', 'error']], how='left', on='ids')
    return 2 * pairs['error'].values, 2 * np.ones(labels.shape[0])

def feval(preds, dataset): # -> eval_name, eval_result, is_bigger_better
    labels = dataset.get_label()
    ids = dataset.ids
    preds[preds<0] = 0
    rmse_per_user = eval_metric(labels, preds, ids)
    eval_name = 'MyRMSE'
    eval_result = rmse_per_user
    is_bigger_better = False
    return eval_name, eval_result, is_bigger_better

def eval_metric(y_true, y_pred, ids):
    pairs = dict(y_true=np.expm1(y_true), y_pred=np.expm1(y_pred), ids=ids)
    pairs = pd.DataFrame(pairs)
    pairs = np.log1p(pairs.groupby('ids').sum()).values
    rmse_per_user = np.mean((pairs[:, 0] - pairs[:, 1])**2)**.5
    return rmse_per_user
%%time
## Model parameters
params = {
    "objective" : "regression", 
    "metric" : "None", #"rmse", 
    "max_depth": 8, 
    "min_child_samples": 20, 
    "reg_alpha": 1, 
    "reg_lambda": 1,
    "num_leaves" : 257, 
    "learning_rate" : 0.01, 
    "subsample" : 0.8, 
    "colsample_bytree" : 0.8, 
    "subsample_freq ": 5
}
folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
## Model
print("model ...")

pred_val = np.zeros(X.shape[0])
pred_test = np.zeros(test_df.shape[0])
eval_hist_li = []
scores = []

dataset = lgb.Dataset(X, y.values, free_raw_data=False)

for fold_n, (train_index, test_index) in enumerate(folds.split(X)):
    print('Fold:', fold_n)
    ids_trn, ids_val = group[train_index], group[test_index]
    dtrain = dataset.subset(train_index)
    dvalid = dataset.subset(test_index)
    dtrain.ids = ids_trn
    dvalid.ids = ids_val
    evals_result = {}
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dtrain, dvalid],
        valid_names=['train','valid'],
        evals_result=evals_result,
        early_stopping_rounds=STOP_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
        feval=feval,
        #fobj=fobj
    )
    eval_hist_li.append(evals_result)
    pred_val[test_index] = model.predict(X.iloc[test_index])
    pred_val[pred_val<0] = 0
    scores.append(eval_metric(y[test_index], pred_val[test_index], ids_val))
    pred_test += model.predict(X_test)
    
pred_test /= N_SPLITS
pred_test[pred_test<0] = 0

np.save('pred_val.npy', pred_val)
np.save('pred_test.npy', pred_test)
print('cv scores', scores)
oof_score = eval_metric(y, pred_val, group)
print('oof_score', oof_score)
# Submission
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = np.expm1(pred_test)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test['PredictedLogRevenue'] = np.log1p(grouped_test['PredictedLogRevenue'])
grouped_test.to_csv(f'sub_{oof_score:.6f}.csv',index=False)
os.listdir('.')