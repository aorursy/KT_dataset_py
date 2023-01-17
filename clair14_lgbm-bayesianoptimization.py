import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h2o
import seaborn as sns
import json
%matplotlib inline
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
def load_df(filename):
    path = "../input/" + filename
    df = pd.read_csv(path, converters={column: json.loads for column in json_cols}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in json_cols:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df
%%time
train_df = load_df('train.csv')
test_df = load_df('test.csv')
test_ind = test_df['fullVisitorId'].copy()
train_df.head()
target = train_df['totals.transactionRevenue'].fillna(0).astype(float)
target = target.apply(lambda x: np.log(x) if x > 0 else x)
del train_df['totals.transactionRevenue']
columns = [col for col in train_df.columns if train_df[col].nunique() > 1]
train_df = train_df[columns].copy()
test_df = test_df[columns].copy()
# some data processing
train_df['date'] = pd.to_datetime(train_df['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
test_df['date'] = pd.to_datetime(test_df['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
useless_columns = ['sessionId', 'visitId']
train_df = train_df.drop(useless_columns, axis = 1)
test_df = test_df.drop(useless_columns, axis = 1)
for col in ['visitNumber', 'totals.hits', 'totals.pageviews']:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)
sns.distplot(target[target!=0])
def feature_design(df):
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
    df['sum_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('sum')

    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')

    df['sum_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')

    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    df['sum_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('sum')
    df['count_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('count')
    df['mean_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('mean')

    df['sum_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('sum')
    df['count_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('count')
    df['mean_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('mean')

    df['user_pageviews_sum'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
    df['user_hits_sum'] = df.groupby('fullVisitorId')['totals.hits'].transform('sum')

    df['user_pageviews_count'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('count')
    df['user_hits_count'] = df.groupby('fullVisitorId')['totals.hits'].transform('count')

    df['user_pageviews_sum_to_mean'] = df['user_pageviews_sum'] / df['user_pageviews_sum'].mean()
    df['user_hits_sum_to_mean'] = df['user_hits_sum'] / df['user_hits_sum'].mean()

    df['user_pageviews_to_region'] = df['user_pageviews_sum'] / df['mean_pageviews_per_region']
    df['user_hits_to_region'] = df['user_hits_sum'] / df['mean_hits_per_region']
    
    return df
%%time
train_df =  feature_design(train_df)
%%time
test_df =  feature_design(test_df)
train_df.head(10)
train_df = train_df.drop(['date', 'visitStartTime', 'fullVisitorId'], axis = 1)
test_df = test_df.drop(['date', 'visitStartTime', 'fullVisitorId'], axis = 1)
cat_cols = train_df.select_dtypes(exclude=['float64', 'int64']).columns
for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
train_df.head()
import lightgbm
from bayes_opt import BayesianOptimization
def rmsle(y, pred):
    assert len(y) == len(pred)
    return np.sqrt(np.mean(np.power(y-pred, 2)))
def lgb_eval(num_leaves,max_depth,lambda_l2,lambda_l1,min_child_samples,bagging_fraction,feature_fraction):
    params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : int(num_leaves),
    "max_depth" : int(max_depth),
    "lambda_l2" : lambda_l2,
    "lambda_l1" : lambda_l1,
    "num_threads" : 4,
    "min_child_samples" : int(min_child_samples),
    "learning_rate" : 0.03,
    "bagging_fraction" : bagging_fraction,
    "feature_fraction" : feature_fraction,
    "subsample_freq" : 5,
    "bagging_seed" : 42,
    "verbosity" : -1
    }
    lgtrain = lightgbm.Dataset(train_df, target,categorical_feature=categorical_features)
    cv_result = lightgbm.cv(params,
                       lgtrain,
                       10000,
                       categorical_feature=categorical_features,
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
    "num_threads" : 4,
    "min_child_samples" : int(min_child_samples),
    "learning_rate" : 0.01,
    "bagging_fraction" : bagging_fraction,
    "feature_fraction" : feature_fraction,
    "subsample_freq" : 5,
    "bagging_seed" : 42,
    "verbosity" : -1
    }
    t_x,v_x,t_y,v_y = train_test_split(train_df, target,test_size=0.2)
    lgtrain = lightgbm.Dataset(t_x, t_y,categorical_feature=categorical_features)
    lgvalid = lightgbm.Dataset(v_x, v_y,categorical_feature=categorical_features)
    model = lightgbm.train(params, lgtrain, 2000, valid_sets=[lgvalid], early_stopping_rounds=100, verbose_eval=100)
    pred_test_y = model.predict(test_df, num_iteration=model.best_iteration)
    return pred_test_y, model
    
def param_tuning(init_points,num_iter,**args):
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (25, 50),
                                                'max_depth': (5, 15),
                                                'lambda_l2': (0.0, 0.05),
                                                'lambda_l1': (0.0, 0.05),
                                                'bagging_fraction': (0.5, 0.8),
                                                'feature_fraction': (0.5, 0.8),
                                                'min_child_samples': (20, 50),
                                                })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
    return lgbBO
categorical_features = list(cat_cols)
result = param_tuning(5,20)
params = result.res['max']['max_params']
params
categorical_features = list(cat_cols)
prediction1,model1 = lgb_train(**params)
prediction2,model2 = lgb_train(**params)
prediction3,model3 = lgb_train(**params)
prediction_lgb = (np.expm1(prediction1)+np.expm1(prediction2)+np.expm1(prediction3))/3
prediction_lgb = [0 if x < 0 else x for x in prediction_lgb]
submission = pd.DataFrame()
submission['fullVisitorId'] = test_ind
submission['PredictedLogRevenue'] = prediction_lgb
submission = submission.groupby('fullVisitorId').sum()['PredictedLogRevenue'].apply(np.log1p).fillna(0).reset_index()
submission.to_csv('submit_lgbm.csv', index=False)
