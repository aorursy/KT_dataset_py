import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
from sklearn.metrics import mean_squared_error
import gc
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GridSearchCV, KFold
gc.enable()
from sklearn.model_selection import train_test_split
#from bayes_opt import BayesianOptimization
train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train.shape, test.shape
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
y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']

for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day
def browser_mapping(x):
    browsers = ['chrome','safari','firefox','internet explorer','edge','opera','coc coc','maxthon','iron']
    if x in browsers:
        return x.lower()
    elif  ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'
    
    
def adcontents_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'
    
def source_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'

train['device.browser'] = train['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
train['trafficSource.adContent'] = train['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
train['trafficSource.source'] = train['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

test['device.browser'] = test['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
test['trafficSource.adContent'] = test['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
test['trafficSource.source'] = test['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

def process_device(data_df):
    print("process device ...")
    data_df['source.country'] = data_df['trafficSource.source'] + '_' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '_' + data_df['trafficSource.medium']
    data_df['browser.category'] = data_df['device.browser'] + '_' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + '_' + data_df['device.operatingSystem']
    return data_df

train = process_device(train)
test = process_device(test)

def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device.deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device.browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device.operatingSystem'] + "_" + data['channelGrouping']
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            data[i + "_" + j] = data[i] + "_" + data[j]
    
    data['content.source'] = data['trafficSource.adContent'] + "_" + data['source.country']
    data['medium.source'] = data['trafficSource.medium'] + "_" + data['source.country']
    return data

train = custom(train)
test = custom(test)
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]

for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])
train.shape, test.shape
train_features = [_f for _f in train.columns if _f not in excluded_features]
# X_train, X_test, y_train, y_test = train_test_split(train[train_features], y_reg, test_size=0.20, random_state=42)
#X_train, y_train = train[train_features], y_reg for Bayesian Optimization
'''
def lgb_eval(num_leaves,max_depth,lambda_l2,lambda_l1,min_child_samples,bagging_fraction,feature_fraction):
    params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : int(num_leaves),
    "max_depth" : int(max_depth),
    "lambda_l2" : lambda_l2,
    "lambda_l1" : lambda_l1,
    "num_threads" : 2,
    "min_child_samples" : int(min_child_samples),
    "learning_rate" : 0.03,
    "bagging_fraction" : bagging_fraction,
    "feature_fraction" : feature_fraction,
    "subsample_freq" : 5,
    "bagging_seed" : 517,
    "verbosity" : -1
    }
    lgtrain = lgb.Dataset(X_train, label=np.log1p(y_train.apply(lambda x : 0 if x < 0 else x)))
    cv_result = lgb.cv(params,
                       lgtrain,
                       1500,
                       categorical_feature=categorical_features,#category_features,
                       early_stopping_rounds=100,
                       stratified=False,
                       nfold=5)
    return -cv_result['rmse-mean'][-1]

def lgb_train(num_leaves,max_depth,lambda_l2,lambda_l1,min_child_samples,bagging_fraction,feature_fraction):
    params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : int(num_leaves),
    "max_depth" : int(max_depth),
    "lambda_l2" : lambda_l2,
    "lambda_l1" : lambda_l1,
    "num_threads" : 2,
    "min_child_samples" : int(min_child_samples),
    "learning_rate" : 0.03,
    "bagging_fraction" : bagging_fraction,
    "feature_fraction" : feature_fraction,
    "subsample_freq" : 5,
    "bagging_seed" : 517,
    "verbosity" : -1
    }
    t_x,v_x,t_y,v_y = train_test_split(X_train,y_train,test_size=0.2)
    lgtrain = lgb.Dataset(t_x, label=np.log1p(t_y.apply(lambda x : 0 if x < 0 else x)))
    lgvalid = lgb.Dataset(v_x, label=np.log1p(v_y.apply(lambda x : 0 if x < 0 else x)))
    model = lgb.train(params, lgtrain, 5000, valid_sets=[lgvalid], early_stopping_rounds=100, verbose_eval=100)
    pred_test_y = model.predict(test_x, num_iteration=model.best_iteration)
    return pred_test_y, model
    
def param_tuning(init_points,num_iter,**args):
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (25, 100),
                                                'max_depth': (5, 20),
                                                'lambda_l2': (0.0, 0.5),
                                                'lambda_l1': (0.0, 0.5),
                                                'bagging_fraction': (0.1, 0.99),
                                                'feature_fraction': (0.1, 0.99),
                                                'min_child_samples': (20, 50),
                                                })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
    return lgbBO

result = param_tuning(8, 22)
result.res['max']['max_params']
'''
#extracted params
params={'learning_rate': 0.01,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 100,
        'verbose': 1,
        'bagging_fraction': 0.99,
        'feature_fraction': 0.99,
        "random_state":517,
        'max_depth': 20,
        "bagging_seed" : 517,
        "verbosity" : -1,
        "bagging_frequency" : 5,
        'lambda_l2': 0,
        'lambda_l1': 0.5,
        'min_child_samples': 20
       }
folds = get_folds(df=train, n_splits=10)

train_features = [_f for _f in train.columns if _f not in excluded_features]

oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    print("Fold:",fold_)
    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
    reg = lgb.LGBMRegressor(**params,
         n_estimators=1500
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
train['predictions'] = np.expm1(oof_reg_preds)
test['predictions'] = sub_reg_preds

trn_data = train[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()

trn_pred_list = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_feats = trn_all_predictions.columns
trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
del trn_data, trn_all_predictions
gc.collect()
full_data.shape
sub_pred_list = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
sub_data = test[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
for f in trn_feats:
    if f not in sub_all_predictions.columns:
        sub_all_predictions[f] = np.nan
sub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))
sub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))
sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)
sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))
sub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)
sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
del sub_data, sub_all_predictions
gc.collect()
sub_full_data.shape
train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
del train, test, trn_pred_list, sub_pred_list
gc.collect()
full_columns = full_data.columns
categorical_features = list(set(categorical_features).intersection(set(full_columns)))
numerical_features = list(set(full_columns).difference(set(categorical_features)))
full_data[categorical_features] = full_data[categorical_features].fillna(-1)
sub_full_data[categorical_features] = sub_full_data[categorical_features].fillna(-1)
full_data[numerical_features] = full_data[numerical_features].fillna(0)
sub_full_data[numerical_features] = sub_full_data[numerical_features].fillna(0)
for num, category_name in enumerate(categorical_features):
    unique, counts = np.unique(full_data[category_name], return_counts=True)
    counts = (counts / counts.sum()).astype(np.float32)
    for i in range(unique.shape[0]):
        full_data.loc[full_data.loc[:, category_name] == unique[i], category_name] = counts[i]
        sub_full_data.loc[sub_full_data.loc[:, category_name] == unique[i], category_name] = counts[i]
    print('Column %i (of %i) processed!'%(num + 1, len(categorical_features)))
X_train = full_data.as_matrix().astype(np.float32)
X_test = sub_full_data[full_data.columns].as_matrix().astype(np.float32)
y_train = trn_user_target.as_matrix().astype(np.float32).ravel()
submission = pd.DataFrame({'PredictedLogRevenue': np.zeros(len(sub_full_data.index))},
                          index=sub_full_data.index)
del full_data, sub_full_data, trn_user_target
gc.collect()
n_folds = 20
sub_preds = np.zeros(X_test.shape[0])
reg = lgb.LGBMRegressor(**params,
                        n_estimators=1500)
kf = KFold(n_splits=n_folds, random_state=517, shuffle=True)
for train_idx, test_idx in kf.split(X_train):
    reg.fit(X_train[train_idx], np.log1p(y_train[train_idx]),
            eval_set=[(X_train[train_idx], np.log1p(y_train[train_idx])),
                      (X_train[test_idx], np.log1p(y_train[test_idx]))],
            eval_names=['TRAIN', 'VALID'],
            early_stopping_rounds=50,
            eval_metric='rmse',
            verbose=101)
    _preds = reg.predict(X_test, num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_preds += _preds / n_folds
    print('Iteration completed!')
submission['PredictedLogRevenue'] = sub_preds
submission[['PredictedLogRevenue']].to_csv('submission.csv', index=True)