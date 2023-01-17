import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pandas.io.json import json_normalize
import lightgbm as lgbm
import time

%matplotlib inline
%load_ext autoreload
%autoreload 2
#Let's define some utilities

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    print(f"read from file '{csv_path}'...")
    df = pd.read_csv(csv_path,
                     converters={
                         column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)
    print("convert columns from json format to plain text...")
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [
            f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(
            column_as_df, right_index=True, left_index=True)
    print(f"Loaded data from '{os.path.basename(csv_path)}'. Shape: {df.shape}")
    return df


def count_unique(df):
    for column in df.columns:
            print(column, len(set(df[column])))
            
def get_list_of_dummies(df):
    result = []
    for column in df.columns:
        if len(set(df[column])) == 1:
            result.append(column)
            
    return result


def get_list_of_values_for_columns_with_nulls(df):
    result = {}
    for column in df.columns:
        s = set(df[column])
        if "null" in s:
            if len(s) < 10:
                print(column, s)
            else:
                print(column, " is ", len(s))
            result[column] = s
            
        
            
    return result

def check_values_of_cat_features(train, test):
    result = {}
    for column in train.columns:
        assert train[column].dtype == test[column].dtype
        if train[column].dtype != np.object:
            continue
        train_set = set(train[column])
        test_set = set(test[column])
        result[column] = test_set - train_set
        if not (test_set <= train_set):
            
            print("For column", column)
#             diff = test_set - train_set
#             if len(diff) < 100:
#                 print(diff)

    return result

def transform_replace_value(df, mapping):
    for column in df.columns:
        if column in mapping:
            cur = mapping[column]
            
            if isinstance(cur, dict):
                df[column] = df[column].apply(cur['func'])
                df[column] = df[column].astype(cur['dtype'])
            else:
                df[column] = df[column].apply(cur)
            
    
    return df

def drop_rare_categories(train, test, min_freq, min_ratio):
    feature_values = {}
    columns = train.columns
    for column in columns:
        assert train[column].dtype == test[column].dtype
        if train[column].dtype != np.object:
            continue
#         print("Processing ", column)
    
        values, counts = np.unique(train[column], return_counts=True)

        argsort = (-counts).argsort()
        counts = counts[argsort]
        values = values[argsort]

        if len(values) < min_freq:
            continue
        
        values_to_spare = set(values[counts >= min_ratio * counts[0]])
        
        train[column] = train[column].apply(lambda v: v if v in values_to_spare else "__rare__")
        test[column] = test[column].apply(lambda v: v if v in values_to_spare else "__rare__")
        
        feature_values[column] = values_to_spare | set(['__rare__'])
        
    
    return train, test, feature_values
        
        
def print_value_counts(train):
    for column in train.columns:
        if train[column].dtype != np.object:
            continue
        print(column, len(set(train[column])))
    
    

def transform_add_is_present_flag(series, null_replacement, res_type=None):
    if res_type is None:
        res_type = series.dtype
    is_present = (series != "null").astype(np.int64)
    series = series.apply(lambda x: x if x != "null" else null_replacement).astype(res_type)
    
    return series, is_present



def plot_countplot(df, column):
    plt.figure(figsize=(15, 15))
    sns.countplot(data=df, y=column, order=df[column].value_counts().index, orient='h')
    
def plot_countplot_series(series):
    plt.figure(figsize=(15, 15))
    sns.countplot(y=series, order=series.value_counts().index, orient='h')
    

    
from sklearn.preprocessing import OneHotEncoder
def transform_categories(train, test):
    transformers = {}
    columns = train.columns
    values_train = []
    values_test = []
    
    cat_columns = []
    for column in columns:
        assert train[column].dtype == test[column].dtype
        if train[column].dtype != np.object:
            continue
        cat_columns.append(column)
#         print("Processing ", column)
        
        transformer = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)
        cur_train = train[column].values.reshape(-1, 1)
        cur_test = test[column].values.reshape(-1, 1)
        train = train.drop(column, axis=1)
        test = test.drop(column, axis=1)
        values_train.append(transformer.fit_transform(cur_train))
        values_test.append(transformer.transform(cur_test))
        
        transformers[column] = transformer
        
    
    feature_names = list(map(lambda x: "num:" + x, train.columns))
    back_feature_names = {}
    
    for name in cat_columns:  
        for feature in transformers[name].categories_[0]:
            current_name = name + ":cat:" + feature
            back_feature_names[current_name] = len(feature_names)
            feature_names.append(current_name)
            
            
    
    values_train = [train.values.astype(np.float32)] + values_train
    values_test = [test.values.astype(np.float32)] + values_test
    
    x_train = np.concatenate(values_train, axis=1)
    x_test = np.concatenate(values_test, axis=1)
    
    return x_train, x_test, feature_names, back_feature_names



def metric(true, pred):
    assert set(true.keys()) == set(pred.keys())
    
    diff = []
    for key in true.keys():
        diff.append(true[key] - pred[key])
    diff = np.array(diff)
#     print(diff)
    result = (sum(diff**2)/len(diff))**0.5
    return result


from collections import defaultdict
def make_prediction(users, predictions):
    assert len(users) == len(predictions), (len(users), len(predictions))
    result = defaultdict(list)
    for user_id, pred in zip(users, predictions):
        result[user_id].append(pred)
    
    for user_id in result:
        result[user_id] = np.log(np.sum(result[user_id]) + 1)
        
    return result


def predict(X, cls, reg):
    ind = cls.predict(X).astype(np.bool)
    
    result = np.zeros(shape=(len(X), ))
    
    
    pr = reg.predict(X[ind])
    result[ind] = np.maximum(np.exp(pr) - 1, 0.0)
    
    return result


def predict_reg(X, reg):
    
    result = np.maximum(np.exp(reg.predict(X)) - 1, 0.0)
    
    return result


def make_result_df(test_pred):
    return pd.DataFrame(list(test_pred.items()), columns=['fullVisitorId','PredictedLogRevenue'])

def write_df(df, file):
    df.sort_values(by='fullVisitorId').to_csv(file, index=False, header=True)
    
        
train = load_df("../input/train.csv")
test = load_df("../input/test.csv")
#define the data preprocessing pipeline:
import datetime
from datetime import datetime
import time
import copy


#used: https://www.kaggle.com/fabiendaniel/lgbm-starter 
#used: https://www.kaggle.com/ashishpatel26/updated-bayesian-lgbm-xgb-cat-fe-kfold-cv
#used: https://www.kaggle.com/ashishpatel26/1-43-plb-feature-engineering-best-model-combined


def null_to_0(value):
    if value != "null":
        return value
    return 0

def replace_value(value, mapping):
    if value in mapping:
        return mapping[value]
    return value


to_int = {"func":null_to_0, "dtype":np.int64}
mapping = {
    "totals.bounces":to_int,
    "totals.newVisits":to_int,
    "trafficSource.adwordsClickInfo.isVideoAd":to_int,
    "trafficSource.isTrueDirect":to_int,
    "totals.hits":to_int,
    "trafficSource.adwordsClickInfo.adNetworkType":lambda value: replace_value(value, {"Content":"null"}),
    "trafficSource.adwordsClickInfo.slot":lambda value: replace_value(value, {'Google Display Network':"null"})
}


def extract_presence_flags(df):
    name = 'totals.pageviews'
    tmp = transform_add_is_present_flag(df[name], 0, res_type=np.int64)
    df[name] = tmp[0]
    df[name + '.is_present'] = tmp[1]
    
    return df

def drop_ids(df):
    columns = ['fullVisitorId', 'sessionId', 'visitId']
    ids = df[columns]
    return df.drop(columns, axis=1), ids

def factorize(train, test, params={}):
    columns = filter(lambda x: train[x].dtype == np.object, train.columns)
    indexers = {}
    for column in columns:
        assert train[column].dtype == test[column].dtype
        train[column], indexer = pd.factorize(train[column])
        test[column] = indexer.get_indexer(test[column])
        indexers[column] = indexer
    
    return train, test, indexers

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
    data_df['date'] = data_df['date'].astype(str)
    return data_df

# def process_format(data_df):
#     print("process format ...")
#     for col in ['visitNumber', 'totals.hits', 'totals.pageviews']:
#         data_df[col] = data_df[col].astype(float)
#     data_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
#     data_df['trafficSource.isTrueDirect'].fillna(False, inplace=True)
#     return data_df
    
def process_device(data_df):
    print("process device ...")
    data_df['browser_category'] = data_df['device.browser'] + '_' + data_df['device.deviceCategory']
    data_df['browser_operatingSystem'] = data_df['device.browser'] + '_' + data_df['device.operatingSystem']
    data_df['source_country'] = data_df['trafficSource.source'] + '_' + data_df['geoNetwork.country']
    return data_df

def process_totals(data_df):
    print("process totals ...")
    data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
    data_df['totals.hits'] = np.log1p(data_df['totals.hits'])
    data_df['totals.pageviews'] = np.log1p(data_df['totals.pageviews'].fillna(0))
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('var')
    return data_df

def process_geo_network(data_df):
    print("process geo network ...")
    data_df['sum_pageviews_per_network_domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    data_df['count_pageviews_per_network_domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    data_df['mean_pageviews_per_network_domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
    data_df['sum_hits_per_network_domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    data_df['count_hits_per_network_domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    data_df['mean_hits_per_network_domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')
    return data_df



def add_features(df, params={}):
    df = process_date_time(df)
    df = process_device(df)
    df = process_totals(df)
    df = process_geo_network(df)


    
    return df



def transform_dataset(train, test, params):
    
    train = copy.deepcopy(train)
    test = copy.deepcopy(test)
    
    feature_info = {"params":copy.deepcopy(params)}
    train = train.fillna("null")
    test = test.fillna("null")
    columns_to_drop_test = get_list_of_dummies(train)
    if 'trafficSource.campaignCode' in test.columns:
        columns_to_drop_test += ['trafficSource.campaignCode']
    else:
        columns_to_drop_train = columns_to_drop_test + ['trafficSource.campaignCode']
        
    test = test.drop(columns_to_drop_test, axis=1)
    train = train.drop(columns_to_drop_train, axis=1)
    
    feature_info['columns_to_drop_train'] = columns_to_drop_train
    feature_info['columns_to_drop_test'] = columns_to_drop_test
    
    train = transform_replace_value(train, mapping)
    test = transform_replace_value(test, mapping)
    
    train = extract_presence_flags(train)
    test = extract_presence_flags(test)
    
    
        
    y_train = transform_add_is_present_flag(train['totals.transactionRevenue'], 0, res_type=np.int64)
#     y_train = list(map(lambda p:p.values, y_train))
    train = train.drop('totals.transactionRevenue', axis=1)
    
    
    train = add_features(train, params)
    test = add_features(test, params)
    
    train, train_ids = drop_ids(train)
    test, test_ids = drop_ids(test) 
    

    
#     train, test, feature_values = utl.drop_rare_categories(train, test, params['min_freq'], params['min_rate'])
#     feature_info['cat_feature_values'] = feature_values
    
    
    train, test, indexers = factorize(train, test, params)
    
    feature_info['indexers'] = indexers
    feature_info['feature_names'] = copy.deepcopy(train.columns)
    
#     train = train.values
#     test = test.values
    
    
    return train, y_train, test, train_ids, test_ids, feature_info
from sklearn.model_selection import GroupKFold

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


train, y_train, test, train_ids, test_ids, feature_info = transform_dataset(train, test,  params={})
train.shape
test.shape
len(feature_info['feature_names'])
feature_info['feature_names']
# train['weekday']
y_train[0].shape
y_train[1].shape
folds = get_folds(df=train_ids, n_splits=5)


param = {'num_leaves': 300,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 1,
         "verbosity": 1}

train_df = train
test_df = test
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
start = time.time()
features = list(train_df.columns)
feature_importance_df = pd.DataFrame()

categorical_feats = list(feature_info['indexers'].keys())
train.dtypes
lgb = lgbm
regressors = []

for fold_, (trn_idx, val_idx) in enumerate(folds):
    trn_data = lgb.Dataset(train_df.iloc[trn_idx], label=np.log(y_train[0].iloc[trn_idx] + 1), categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_df.iloc[val_idx], label=np.log(y_train[0].iloc[val_idx] + 1), categorical_feature=categorical_feats)
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    regressors.append(clf)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_df, num_iteration=clf.best_iteration) / len(folds)
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(clf, max_num_features=50, height=0.8, ax=ax)
lgb_features = clf.feature_importance
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=20)
plt.show()
test_pred = make_prediction(test_ids['fullVisitorId'], np.maximum(np.exp(predictions) - 1, 0))
res_p = make_result_df(test_pred)
write_df(res_p, './prediction.csv')

