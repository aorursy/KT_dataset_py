import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h2o
import seaborn as sns
import json
%matplotlib inline
from pandas.io.json import json_normalize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)
def load_df(filename):
    json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(filename, converters={column: json.loads for column in json_cols}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in json_cols:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df
%%time
train_df = load_df('../input/train.csv')
test_df = load_df('../input/test.csv')
train_ind = train_df['fullVisitorId'].copy()
test_ind = test_df['fullVisitorId'].copy()
def feature_design(df):
    df['date'] = pd.to_datetime(df['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
    for col in ['visitNumber', 'totals.hits', 'totals.pageviews']:
        df[col] = df[col].astype(float)
        
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['weekofyear'] = df['date'].dt.weekofyear
    
    df['month_unique_user_count'] = df.groupby('month')['fullVisitorId'].transform('nunique')
    df['day_unique_user_count'] = df.groupby('day')['fullVisitorId'].transform('nunique')
    df['weekday_unique_user_count'] = df.groupby('weekday')['fullVisitorId'].transform('nunique')
    df['weekofyear_unique_user_count'] = df.groupby('weekofyear')['fullVisitorId'].transform('nunique')
    
    df['browser_category'] = df['device.browser'] + '_' + df['device.deviceCategory']
    df['browser_operatingSystem'] = df['device.browser'] + '_' + df['device.operatingSystem']
    df['source_country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']
    
    df['visitNumber'] = np.log1p(df['visitNumber'])
    df['totals.hits'] = np.log1p(df['totals.hits'])
    df['totals.pageviews'] = np.log1p(df['totals.pageviews'].fillna(0))
    
    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')
    
    df['mean_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('mean')
    df['sum_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('median')

    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('median')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')

    df['sum_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('median')
    df['mean_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')

    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('median')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    df['sum_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('sum')
    df['count_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('median')
    df['mean_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('mean')

    df['sum_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('sum')
    df['count_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('median')
    df['mean_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('mean')

    df['user_pageviews_sum'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
    df['user_hits_sum'] = df.groupby('fullVisitorId')['totals.hits'].transform('sum')

    df['user_pageviews_count'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('count')
    df['user_hits_count'] = df.groupby('fullVisitorId')['totals.hits'].transform('count')

    df['user_pageviews_sum_to_mean'] = df['user_pageviews_sum'] / df['user_pageviews_sum'].mean()
    df['user_hits_sum_to_mean'] = df['user_hits_sum'] / df['user_hits_sum'].mean()

    df['user_pageviews_to_region'] = df['user_pageviews_sum'] / df['mean_pageviews_per_region']
    df['user_hits_to_region'] = df['user_hits_sum'] / df['mean_hits_per_region']
    
    useless_columns = ['sessionId', 'visitId', 'fullVisitorId', 'date', 'visitStartTime','user_pageviews_sum', 'user_hits_sum',
                      'user_pageviews_count', 'user_hits_count']
    df = df.drop(useless_columns, axis = 1)
    
    return df
target = train_df['totals.transactionRevenue'].fillna(0).astype(float)
target = target.apply(lambda x: np.log(x) if x > 0 else x)
del train_df['totals.transactionRevenue']

columns = [col for col in train_df.columns if train_df[col].nunique() > 1]

train_df = train_df[columns].copy()
test_df = test_df[columns].copy()

train_df =  feature_design(train_df)
test_df =  feature_design(test_df)
cat_cols = train_df.select_dtypes(exclude=['float64', 'int64']).columns

for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
import lightgbm
def lgb_train(X, y, test_X, params):
    kf = KFold(n_splits=5, shuffle=True, random_state=2017)
    pred_test = 0
    pred_train = 0
    for dev_index, val_index in kf.split(X):
        train_x, valid_x = X.iloc[dev_index,:], X.iloc[val_index,:]
        train_y, valid_y = y[dev_index], y[val_index]
        lgtrain = lightgbm.Dataset(train_x, train_y,categorical_feature=list(cat_cols))
        lgvalid = lightgbm.Dataset(valid_x, valid_y,categorical_feature=list(cat_cols))
        model = lightgbm.train(params, lgtrain, 2000, valid_sets=[lgvalid], early_stopping_rounds=100, verbose_eval=100)
        pred_test_iter = model.predict(test_X, num_iteration=model.best_iteration)
        pred_test_iter[pred_test_iter<0]=0
        pred_test+=pred_test_iter
        pred_train_iter = model.predict(X, num_iteration=model.best_iteration)
        pred_train_iter[pred_train_iter<0]=0
        pred_train+=pred_train_iter
    pred_test /= 5.
    pred_train  /= 5.
    return pred_test, pred_train
params_lgb = {'objective': 'regression', 
          'metric': 'rmse', 
          'num_leaves': 49, 
          'max_depth': 14, 
          'lambda_l2': 0.01931081461346337, 
          'lambda_l1': 0.007163878762237125, 
          'num_threads': 4, 
          'min_child_samples': 40, 
          'learning_rate': 0.01, 
          'bagging_fraction': 0.7910460446769023, 
          'feature_fraction': 0.5046791892199741, 
          'subsample_freq': 5, 
          'bagging_seed': 42, 
          'verbosity': -1}
sub_lgb_test, sub_lgb_train = lgb_train(train_df, target, test_df, params_lgb)
import xgboost as xgb
def xgb_train(X, y, test_X, params):
    kf = KFold(n_splits=5, shuffle=True, random_state=2017)
    pred_test_xgb = 0
    pred_train_xgb = 0
    for dev_index, val_index in kf.split(train_df):
        train_x, valid_x = X.loc[dev_index,:], X.loc[val_index,:]
        train_y, valid_y = y[dev_index], y[val_index]
        xgb_train_data = xgb.DMatrix(train_x, train_y)
        xgb_val_data = xgb.DMatrix(valid_x, valid_y)
        xgb_submit_data = xgb.DMatrix(test_X)
        xgb_submit_data_train = xgb.DMatrix(X)
        xgb_model = xgb.train(params, xgb_train_data, 
                          num_boost_round=2000, 
                          evals= [(xgb_train_data, 'train'), (xgb_val_data, 'valid')],
                          early_stopping_rounds=100, 
                          verbose_eval=500
                         )
        pred_test = xgb_model.predict(xgb_submit_data, ntree_limit=xgb_model.best_ntree_limit)
        pred_train = xgb_model.predict(xgb_submit_data_train, ntree_limit=xgb_model.best_ntree_limit)
        pred_test[pred_test<0]=0
        pred_train[pred_train<0]=0
        pred_test_xgb += pred_test
        pred_train_xgb += pred_train
    pred_test_xgb /= 5.
    pred_train_xgb /= 5.
    return pred_test_xgb, pred_train_xgb
params_xgb = {
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'eta': 0.001,
            'max_depth': 7,
            'gamma': 1.3250360141843498, 
            'min_child_weight': 13.0958516960316, 
            'max_delta_step': 8.88492863796954, 
            'subsample': 0.9864199446951019, 
            'colsample_bytree': 0.8376539278239742,
            'subsample': 0.6,
            'colsample_bytree': 0.8,
            'alpha':0.001,
            "num_leaves" : 40,
            'random_state': 42,
            'silent': True,
            }
sub_xgb_test, sub_xgb_train = xgb_train(train_df, target, test_df, params_xgb)
from catboost import CatBoostRegressor
def cat_train(X, y, test_X):
    kf = KFold(n_splits=5, shuffle=True, random_state=2017)
    pred_test_cat = 0
    pred_train_cat = 0
    for dev_index, val_index in kf.split(train_df):
        train_x, valid_x = X.loc[dev_index,:], X.loc[val_index,:]
        train_y, valid_y = y[dev_index], y[val_index]
        model = CatBoostRegressor(iterations=1000,
                             learning_rate=0.05,
                             depth=10,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
        model.fit(train_x, train_y, eval_set=(valid_x, valid_y),use_best_model=True,verbose=True, 
                  cat_features= [i for i in range(len(train_df.columns)) if train_df.columns[i] in cat_cols])
        pred_test = model.predict(test_X)
        pred_test[pred_test<0]=0
        pred_test_cat += pred_test
        pred_train = model.predict(X)
        pred_train[pred_train<0]=0
        pred_train_cat += pred_train
    pred_test_cat /= 5.
    pred_train_cat /= 5.
    return pred_test_cat, pred_train_cat
sub_cat_test, sub_cat_train = cat_train(train_df, target, test_df)
last = pd.DataFrame()
last['fullVisitorId'] = train_ind
last['lgbm'] = sub_lgb_train
last['xbm'] = sub_xgb_train
last['cat'] = sub_cat_train
last_test = pd.DataFrame()
last_test['fullVisitorId'] = test_ind
last_test['lgbm'] = sub_lgb_test
last_test['xbm'] = sub_xgb_test
last_test['cat'] = sub_cat_test
from sklearn.linear_model import Ridge
model = Ridge().fit(last, target)
pred = model.predict(last_test)
pred[pred<0] = 0
submission = pd.DataFrame()
submission['fullVisitorId'] = test_ind
submission['PredictedLogRevenue'] = pred
submission = submission.groupby('fullVisitorId').sum()['PredictedLogRevenue'].fillna(0).reset_index()
submission.to_csv('submit_stack1.csv', index=False)
