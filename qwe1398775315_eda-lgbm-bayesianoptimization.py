# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize
import json
import matplotlib.pyplot as plt
import lightgbm as lgb
import datetime
import seaborn as sns
from bayes_opt import BayesianOptimization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,train_test_split

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['visitStartTime_'] = pd.to_datetime(df['visitStartTime'],unit="s")
    df['visitStartTime_year'] = df['visitStartTime_'].apply(lambda x: x.year)
    df['visitStartTime_month'] = df['visitStartTime_'].apply(lambda x: x.month)
    df['visitStartTime_day'] = df['visitStartTime_'].apply(lambda x: x.day)
    df['visitStartTime_weekday'] = df['visitStartTime_'].apply(lambda x: x.weekday())
    return df
date_features = [#"year","month","day","weekday",'visitStartTime_year',
    "visitStartTime_month","visitStartTime_day","visitStartTime_weekday"]
%%time
train_df = load_df("../input/train.csv")
test_df = load_df("../input/test.csv")
train_df.head()
train_df.dtypes
test_df.info()
pd.value_counts(train_df.dtypes).plot(kind="bar")
plt.title("type of train data")
def bar_plot(column,**args):
    pd.value_counts(train_df[column]).plot(kind="bar",**args)
    
constant_column = [col for col in train_df.columns if len(train_df[col].unique()) == 1]
print(list(constant_column))
train_df.drop(columns=constant_column,inplace=True)
num_col = ["totals.hits", "totals.pageviews", "visitNumber", 
           'totals.bounces',  'totals.newVisits']
for col in num_col:
    train_df[col] = train_df[col].fillna("0").astype("int32")
    test_df[col] = test_df[col].fillna("0").astype("int32")
train_df.dtypes
train_df.head()
bar_plot("channelGrouping")
bar_plot("device.browser",figsize=(12,10))
bar_plot("device.isMobile")
bar_plot("device.deviceCategory")
new_features = ["hits_per_pageviews"]
new_category_features = ["is_high_hits"]
def feature_engineering(df):
    line = 4
    df['hits_per_pageviews'] = (df["totals.hits"]/(df["totals.pageviews"])).apply(lambda x: 0 if np.isinf(x) else x)
    df['is_high_hits'] = np.logical_or(train_df["totals.hits"]>line,train_df["totals.pageviews"]>line).astype(np.int32)
feature_engineering(train_df)
feature_engineering(test_df)
add_time_features(train_df)
_ = add_time_features(test_df)
category_features = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent",
            #"trafficSource.adContent", 
            #"trafficSource.adwordsClickInfo.adNetworkType", 
            #"trafficSource.adwordsClickInfo.gclId", 
            #"trafficSource.adwordsClickInfo.page", 
            #"trafficSource.adwordsClickInfo.slot",
            #"trafficSource.campaign",
            #"trafficSource.keyword", 
            "trafficSource.medium", 
            #"trafficSource.referralPath", 
            "trafficSource.source",

            #'trafficSource.adwordsClickInfo.isVideoAd',
            'trafficSource.isTrueDirect',
            #"filtered_keyword"
            ] + date_features
target = 'totals.transactionRevenue'
useless_col = ["trafficSource.adContent", 
              "trafficSource.adwordsClickInfo.adNetworkType", 
              "trafficSource.adwordsClickInfo.page",
              "trafficSource.adwordsClickInfo.slot",
              "trafficSource.campaign",
              "trafficSource.referralPath",
              'trafficSource.adwordsClickInfo.isVideoAd',
              "trafficSource.adwordsClickInfo.gclId",
              "trafficSource.keyword"]
train_df.head()
useless_df = train_df[useless_col]
useless_df.info()
for col in useless_col:
    print("-"*10,col,"-"*10)
    print("unique value numbers:",len(useless_df[col].unique()))
    print("null rate:",useless_df[col].isna().sum()/len(useless_df[col]))
for col in category_features:
    print("-"*10,col,"-"*10)
    print("unique value numbers:",len(train_df[col].unique()))
    print("null rate:",train_df[col].isna().sum()/len(train_df[col]))
train_df[target] = train_df[target].fillna("0").astype("int32")
all_features = category_features+num_col+new_features+new_category_features
all_features
# dev_df = train_df[train_df['date']<=pd.to_datetime('20170531', format='%Y%m%d')]
# val_df = train_df[train_df['date']>pd.to_datetime('20170531', format='%Y%m%d')]

# dev_x = dev_df[all_features]
# dev_y = dev_df[target]
# val_x = val_df[all_features]
# val_y = val_df[target]
# test_x = test_df[all_features]
# for col in category_features:
#     print("transform column {}".format(col))
#     lbe = LabelEncoder()
#     lbe.fit(pd.concat([train_df[col],test_x[col]]).astype("str"))
#     dev_x[col] = lbe.transform(dev_x[col].astype("str"))
#     val_x[col] = lbe.transform(val_x[col].astype("str"))
#     test_x[col] = lbe.transform(test_x[col].astype("str"))
train_df["totals.hits"].describe()
sns.distplot(train_df["totals.hits"],kde=False)
train_df["totals.pageviews"].describe()
sns.distplot(train_df["totals.pageviews"],kde=False)
sns.jointplot("totals.pageviews","totals.hits",data=train_df)
sns.jointplot("totals.pageviews",target,data=train_df)
sns.jointplot("totals.hits",target,data=train_df)
line = 4
high_hits_pageviews_df = train_df[np.logical_or(train_df["totals.hits"]>line,train_df["totals.pageviews"]>line)]
low_hits_pageviews_df = train_df[np.logical_and(train_df["totals.hits"]<=line,train_df["totals.pageviews"]<=line)]
print("high rate :",high_hits_pageviews_df.shape[0]/train_df.shape[0])
print("low rate :",low_hits_pageviews_df.shape[0]/train_df.shape[0])
high_hits_pageviews_df[target].describe()
low_hits_pageviews_df[target].describe()
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(16,8))
sns.distplot(high_hits_pageviews_df[target],kde=False,ax=axes[0])
axes[0].set_title("distribution of high hits transactionRevenue")

sns.distplot(low_hits_pageviews_df[target],kde=False,ax=axes[1])
axes[1].set_title("distribution of low hits transactionRevenue")
print("zero rate of transactionRevenue:",(train_df[target]==0).sum()/train_df.shape[0])
print("zero rate of high hits transactionRevenue:",(high_hits_pageviews_df[target]==0).sum()/ high_hits_pageviews_df.shape[0])
print("zero rate of low hits transactionRevenue:",(low_hits_pageviews_df[target]==0).sum()/ low_hits_pageviews_df.shape[0])
train_df["hits_per_pageviews"].describe()
sns.jointplot("hits_per_pageviews",target,data=train_df)
visitStartTime_df = train_df[["visitStartTime",'visitStartTime_year',"visitStartTime_month","visitStartTime_day","visitStartTime_weekday",target]]
visitStartTime_df["visitStartTime"] = pd.to_datetime(visitStartTime_df["visitStartTime"],unit="s")
visitStartTime_df["visitStartDate"] = visitStartTime_df["visitStartTime"].apply(lambda x: x.date())
def plot_dist_date(col,kind="bar"):
    fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
    visitStartTime_df.groupby(col)[target].agg(["sum"]).plot(kind=kind,title="sum of transactionRevenue:"+col,ax=axes[0])
    visitStartTime_df.groupby(col)[target].agg(["count"]).plot(kind=kind,title="count of transactionRevenue:"+col,ax=axes[1])
    plt.tight_layout()
plot_dist_date("visitStartDate",kind="line")
test_visitStartTime_df = test_df[["visitStartTime",'visitStartTime_year',"visitStartTime_month","visitStartTime_day","visitStartTime_weekday"]]
test_visitStartTime_df["visitStartTime"] = pd.to_datetime(test_visitStartTime_df["visitStartTime"],unit="s")
test_visitStartTime_df["visitStartDate"] = test_visitStartTime_df["visitStartTime"].apply(lambda x: x.date())
test_visitStartTime_df.groupby("visitStartDate")["visitStartTime"].agg("count").plot(figsize=(8,6),title="count of test")
plot_dist_date("visitStartTime_year")
plot_dist_date("visitStartTime_month")
plot_dist_date("visitStartTime_day")
plot_dist_date("visitStartTime_weekday")
col = "trafficSource.cleanedkeyword"
train_df[col] = train_df["trafficSource.keyword"].apply(lambda x :x if isinstance(x,float) and np.isnan(x) else x.lower()).apply(lambda x :x if isinstance(x,float) and np.isnan(x) else x.replace("+", ""))
test_df[col] = test_df["trafficSource.keyword"].apply(lambda x :x if isinstance(x,float) and np.isnan(x) else x.lower())
print("-"*10,"train","-"*10)
print("unique value numbers:",len(train_df[col].unique()))
print("null rate:",train_df[col].isna().sum()/len(train_df[col]))
print("-"*10,"test","-"*10)
print("unique value numbers:",len(test_df[col].unique()))
print("null rate:",test_df[col].isna().sum()/len(test_df[col]))
pd.value_counts(train_df[col]).sort_values(ascending=False)[0:20]
pd.value_counts(test_df[col]).sort_values(ascending=False)[0:20]
train_df.groupby(col)[target].agg("sum").sort_values(ascending=False)[0:29].apply(lambda x: np.log1p(x)).plot(kind="bar",figsize=(20,5),title="sum revenue of keyword")
none_zero_keywords= set(train_df.groupby(col)[target].agg("sum").sort_values(ascending=False)[0:28].index)
test_keywords_set = set(test_df[col].unique())
intersection_keyword = none_zero_keywords.intersection(test_keywords_set)
print("len:",len(intersection_keyword))
intersection_keyword
train_df.groupby(col)[target].agg("sum").sort_values(ascending=False)[0:29].apply(lambda x: np.log1p(x)).plot(kind="bar",figsize=(20,5),title="zero revenue of keyword")
def add_keyword_feature(df):
    col_name ="filtered_keyword"
    sets = intersection_keyword.difference({'(automatic matching)',
     '(not provided)',
     '(remarketing/content targeting)'})
    df[col_name] = df[col].apply(lambda x: x if x in sets else "other")
add_keyword_feature(train_df)
add_keyword_feature(test_df)
# no improvement ,something wrong
none_zero_keywords.difference(test_keywords_set)
train_df.groupby(col)[target].agg("count").sort_values(ascending=False)[0:40].apply(lambda x: np.log1p(x)).plot(kind="bar",figsize=(20,5),title="count revenue of keyword")
train_x = train_df[all_features]
train_y = train_df[target]
test_x = test_df[all_features]
for col in category_features:
    print("transform column {}".format(col))
    lbe = LabelEncoder()
    lbe.fit(pd.concat([train_df[col],test_x[col]]).astype("str"))
    train_x[col] = lbe.transform(train_x[col].astype("str"))
    test_x[col] = lbe.transform(test_x[col].astype("str"))
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
    lgtrain = lgb.Dataset(train_x, label=np.log1p(train_y.apply(lambda x : 0 if x < 0 else x)),categorical_feature=category_features)
    cv_result = lgb.cv(params,
                       lgtrain,
                       10000,
                       categorical_feature=category_features,
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
    t_x,v_x,t_y,v_y = train_test_split(train_x,train_y,test_size=0.2)
    lgtrain = lgb.Dataset(t_x, label=np.log1p(t_y.apply(lambda x : 0 if x < 0 else x)),categorical_feature=category_features)
    lgvalid = lgb.Dataset(v_x, label=np.log1p(v_y.apply(lambda x : 0 if x < 0 else x)),categorical_feature=category_features)
    model = lgb.train(params, lgtrain, 2000, valid_sets=[lgvalid], early_stopping_rounds=100, verbose_eval=100)
    pred_test_y = model.predict(test_x, num_iteration=model.best_iteration)
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
result = param_tuning(5,15)
result.res['max']['max_params']
prediction1,model1 = lgb_train(**result.res['max']['max_params'])
prediction2,model2 = lgb_train(**result.res['max']['max_params'])
prediction3,model3 = lgb_train(**result.res['max']['max_params'])
# param = {'num_leaves': 45.61216380347129,
#  'max_depth': 11.578579827303919,
#  'lambda_l2': 0.0107663924764632,
#  'lambda_l1': 0.046541310399201855,
#  'bagging_fraction': 0.7851516324443661,
#  'feature_fraction': 0.7944881085591733,
#  'min_child_samples': 28.5601473698899}
# prediction,model = lgb_train(**param)
test_df['PredictedLogRevenue'] = (np.expm1(prediction1)+np.expm1(prediction2)+np.expm1(prediction3))/3
submit = test_df[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum()['PredictedLogRevenue'].apply(np.log1p).fillna(0).reset_index()
submit.to_csv('submission.csv', index=False)
lgb.plot_importance(model, figsize=(15, 10),height=0.8)
plt.show()
lgb.plot_importance(model, figsize=(15, 10),height=0.8,importance_type="gain")
plt.show()