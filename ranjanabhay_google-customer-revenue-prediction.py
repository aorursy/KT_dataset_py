# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import json
from pandas.io.json import json_normalize
import time
from datetime import datetime
import gc
import psutil
from sklearn.preprocessing import LabelEncoder

columns_to_parse = ['device','geoNetwork','totals','trafficSource']

def parse_dataframe(path):
    #path = '/Users/abhayranjan/kaggle/google/all/train.csv'
    data_df= pd.read_csv(path,converters={column: json.loads for column in columns_to_parse},
                                dtype={'fullVisitorId':str})
    
    #parse the json type columns
    for col in columns_to_parse:
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        # we drop the object column processed and we add the column created from json fields.
        data_df = data_df.drop(col,axis=1).merge(json_col_df,right_index=True,left_index=True)
        
    return data_df 
def process_datetime(data_df):
    data_df['date'] = data_df['date'].astype(str)
    data_df['date'] = data_df['date'].apply(lambda x:x[:4]+"-"+x[4:6]+"-"+x[6:])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df['year'] = data_df['date'].dt.year
    data_df['month'] = data_df['date'].dt.month
    data_df['day'] = data_df['date'].dt.day
    data_df['weekday'] =data_df['date'].dt.weekday
    data_df['weekofyear'] =data_df['date'].dt.weekofyear
    data_df['month_unique_user_count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
    data_df['day_unique_user_count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday_unique_user_count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')
    return data_df
def process_format(data_df):
    print("Inside process_format function")
    for col in ['visitNumber','totals_hits','totals_pageviews']:
        data_df[col] = data_df[col].astype(float)
            
    data_df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True,inplace=True)
    data_df['trafficSource_isTrueDirect'].fillna('False',inplace=True)
    return data_df
def process_device(data_df):
    print("process device")
    data_df['browser_category'] = data_df['device_browser'] + "_"+data_df['device_deviceCategory']
    data_df['browser_os'] = data_df['device_browser'] + "_"+data_df['device_operatingSystem']
    return data_df
def process_totals(data_df):
    print("process totals..")
    data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
    data_df['totals_hits'] = np.log1p(data_df['totals_hits'])
    data_df['totals_pageviews'] = np.log1p(data_df['totals_pageviews']).fillna(0)
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
    data_df['mean_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('mean')
    data_df['sum_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('sum')
    data_df['max_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('max')
    data_df['min_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('min')
    return data_df
def process_geo_network(data_df):
    print("process geo network...")
    data_df['sum_pageviews_per_network_domain'] = data_df.groupby(['geoNetwork_networkDomain'])['totals_pageviews'].transform('sum')
    data_df['count_pageviews_per_network_domain'] = data_df.groupby(['geoNetwork_networkDomain'])['totals_pageviews'].transform('count') 
    data_df['mean_pageviews_per_network_domain'] = data_df.groupby(['geoNetwork_networkDomain'])['totals_pageviews'].transform('mean')
    data_df['sum_hits_per_network_domain'] = data_df.groupby(['geoNetwork_networkDomain'])['totals_hits'].transform('sum')
    data_df['count_hits_per_network_domain'] = data_df.groupby(['geoNetwork_networkDomain'])['totals_hits'].transform('count')
    data_df['mean_hits_per_network_domain'] = data_df.groupby(['geoNetwork_networkDomain'])['totals_hits'].transform('mean')
    return data_df                                                  
pd.set_option('display.max.columns',None)
def process_traffic_source(data_df):
    print("process traffic source....")
    data_df['source_country'] = data_df['trafficSource_source']+"_" +data_df['geoNetwork_country']
    data_df['campaign_medium'] = data_df['trafficSource_campaign']+"_" +data_df['trafficSource_medium']
    data_df['medium_hits_mean'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('mean')
    data_df['medium_hits_min'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('min')
    data_df['medium_hits_max'] =data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('max')
    data_df['medium_hits_sum'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('sum')
    return data_df
g_train_df = parse_dataframe('../input/train.csv')
g_train_df = process_datetime(g_train_df)
g_test_df = parse_dataframe('../input/test.csv')
g_test_df = process_datetime(g_test_df) 

cols_to_drop = [col for col in g_train_df.columns if g_train_df[col].nunique(dropna=False) == 1]
g_train_df.drop(cols_to_drop, axis=1, inplace=True)
g_test_df.drop([col for col in cols_to_drop if col in g_test_df.columns], axis=1, inplace=True)

###only one not null value
g_train_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)

###converting columns format
g_train_df['totals_transactionRevenue'] = g_train_df['totals_transactionRevenue'].astype(float)
g_train_df['totals_transactionRevenue'] = g_train_df['totals_transactionRevenue'].fillna(0)

g_train_df = process_format(g_train_df)
g_tain_df = process_device(g_train_df)
g_train_df = process_totals(g_train_df)
g_train_df = process_geo_network(g_train_df)
g_train_df = process_traffic_source(g_train_df)   

g_test_df = process_format(g_test_df)
g_test_df = process_device(g_test_df)
g_test_df = process_totals(g_test_df)
g_test_df = process_geo_network(g_test_df)
g_test_df = process_traffic_source(g_test_df)   

# Numeric Columns

num_cols = ['month_unique_user_count','day_unique_user_count','weekday_unique_user_count','visitNumber','total_hits','total_pageviews','mean_hits_per_day','sum_hits_per_day','max_hits_per_day','min_hits_per_day','var_hits_per_day',
           'mean_pageviews_per_day','sum_pageviews_per_day','max_pageviews_per_day','min_pageviews_per_day',
           'sum_pageviews_per_network_domain','count_pageviews_per_network_domain','mean_pageviews_per_network_domain','sum_hits_per_network_domain',
            'count_hits_per_network_domain','mean_hits_per_network_domain','medium_hits_mean',
           'medium_hits_min','medium_hits_max','medium_hits_sum']

not_used_cols = ['visitNumber','date','fullVisitorId','sessionId','visitId','visitStartTime',
                'totals_transactionRevenue','trafficSource_referralPath']

cat_cols = [col for col in g_train_df.columns if col not in num_cols and col not in not_used_cols]
print(cat_cols)

merged_df = pd.concat([g_train_df,g_test_df])
ohe_columns = []
for i in cat_cols:
    if len(set(merged_df[i].values))<100:
        ohe_columns.append(i)
            
print('ohe_cols:',ohe_columns)
#print(len(ohe_columns))
trn_shape=g_train_df.shape[0]
merged_df = pd.get_dummies(merged_df,columns=ohe_columns)
g_train_df = merged_df[:trn_shape]
g_test_df  = merged_df[trn_shape:]
g_train_df = g_train_df.loc[:,~g_train_df.columns.duplicated()]
g_test_df = g_test_df.loc[:,~g_test_df.columns.duplicated()]
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    if col in ohe_columns:
        continue
    lbl = LabelEncoder()
    lbl.fit(list(g_train_df[col].values.astype('str')) + list(g_test_df[col].values.astype('str')))
    g_train_df[col]= lbl.transform(list(g_train_df[col].values.astype('str')))
    g_test_df[col]= lbl.transform(list(g_test_df[col].values.astype('str')))      
    
X = g_train_df.drop(not_used_cols,axis=1)
y = g_train_df['totals_transactionRevenue']

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import model_selection,preprocessing,metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Stratified Sampling
FOLDs= StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
oof_lgb = np.zeros(len(g_train_df))
predictions_lgb = np.zeros(len(g_test_df))
#print(list(X.columns))                          
features_lgb = list(X.columns)
feature_importance_df_lgb = pd.DataFrame()
                           
for fold_, (trn_idx,val_idx) in enumerate(FOLDs.split(X,y_categorized)):
    trn_data = lgb.Dataset(X.iloc[trn_idx],label=y_log.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[val_idx],label=y_log.iloc[val_idx])
    num_round=2000
    clf = lgb.train(lgb_params1,trn_data,num_round,valid_sets=[trn_data,val_data],verbose_eval=1000,early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X.iloc[val_idx],num_iteration=clf.best_iteration)
    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb,fold_importance_df_lgb],axis=0)
    predictions_lgb += clf.predict(X_test,num_iteration=clf.best_iteration) / FOLDs.n_splits
    
    
cols = feature_importance_df_lgb[["feature","importance"]].groupby("feature").mean().sort_values(by="importance",ascending=False)[:50].index
best_features_lgb = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]
plt.figure(figsize=(14,10))
sns.barplot(x="importance",y="feature",data=best_features_lgb.sort_values(by="importance",ascending=False))
plt.title('LightGBM features( avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importancers.png')
x = []
for i in oof_lgb:
    if i<0:
        x.append(0.0)
    else:
        x.append(i)
cv_lgb = mean_squared_error(x,y_log)**0.5
cv_lgb = str(cv_lgb)
cv_lgb = cv_lgb[:10]
pd.DataFrame({'preds':x}).to_csv('lgb_oof_'+cv_lgb + '.csv',index=False)
print("CV_LGB:",cv_lgb)

sub_df = g_test_df[['fullVisitorId']].copy()
predictions_lgb[predictions_lgb<0] = 0
sub_df['PredictedLogRevenue'] = np.expm1(predictions_lgb)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ['fullVisitorId',"PredictedLogRevenue"]

sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("submission.csv",index=False)

