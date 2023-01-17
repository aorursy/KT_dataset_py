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
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import GroupKFold
df_train = pd.read_csv(filepath_or_buffer="../input/train.csv",dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
df_test = pd.read_csv(filepath_or_buffer="../input/test.csv",dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
df_train = df_train.drop(["date", "sessionId","socialEngagementType", "visitId"],axis = 1)
df_test = df_test.drop(["date", "sessionId","socialEngagementType", "visitId"],axis = 1)
df_train.shape, df_test.shape
totals_columns = ["transactionRevenue", "newVisits", "bounces", "pageviews", "hits"]
tmp_totals_df = pd.DataFrame(df_train.totals.apply(json.loads).tolist())[totals_columns]
df_train = pd.concat([df_train,tmp_totals_df] , axis=1)

totals_columns.remove("transactionRevenue")
tmp_totals_df = pd.DataFrame(df_test.totals.apply(json.loads).tolist())[totals_columns]
df_test = pd.concat([df_test,tmp_totals_df],axis = 1 )
df_train["transactionRevenue"] = df_train.transactionRevenue.fillna(0.0)
df_train = df_train.drop(["totals"],axis=1)
df_test = df_test.drop(["totals"],axis=1)
del tmp_totals_df
df_train.shape, df_test.shape
geo_columns = ["continent","subContinent","country", "city","region", "metro" , "networkDomain"]
tmp_geo_df = pd.DataFrame(df_train.geoNetwork.apply(json.loads).tolist())[geo_columns]
df_train = pd.concat([df_train, tmp_geo_df],axis=1)

tmp_geo_df = pd.DataFrame(df_test.geoNetwork.apply(json.loads).tolist())[geo_columns]
df_test = pd.concat([df_test, tmp_geo_df],axis=1)
del tmp_geo_df
df_train = df_train.drop(["geoNetwork"],axis=1)
df_test = df_test.drop(["geoNetwork"],axis=1)
df_train.shape, df_test.shape
devices_columns = ["browser","deviceCategory","isMobile","operatingSystem"]
tmp_device_df = pd.DataFrame(df_train.device.apply(json.loads).tolist())[devices_columns]
df_train=pd.concat([df_train,tmp_device_df],axis = 1)

tmp_device_df = pd.DataFrame(df_test.device.apply(json.loads).tolist())[devices_columns]
df_test=pd.concat([df_test,tmp_device_df],axis = 1)
del tmp_device_df 
df_train = df_train.drop(["device"], axis = 1)
df_test = df_test.drop(["device"], axis = 1)
df_train.shape, df_test.shape
trafficSource_columns = ["campaign","medium" , "source","adContent","isTrueDirect", "keyword","referralPath"]
tmp_traffic_df = pd.DataFrame(df_train.trafficSource.apply(json.loads).tolist())[trafficSource_columns]
df_train = pd.concat([df_train,tmp_traffic_df] , axis = 1 )

tmp_traffic_df = pd.DataFrame(df_test.trafficSource.apply(json.loads).tolist())[trafficSource_columns]
df_test = pd.concat([df_test,tmp_traffic_df] , axis = 1 )
del tmp_traffic_df 
df_train = df_train.drop(["trafficSource"] , axis = 1 )
df_test = df_test.drop(["trafficSource"] , axis = 1 )
df_train.shape , df_test.shape
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)
for df in [df_train, df_test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour_of_day'] = df['date'].dt.hour
    df['day_of_month'] = df['date'].dt.day
df_train = df_train.drop(["date","visitStartTime"],axis=1)
df_test = df_test.drop(["date","visitStartTime"],axis=1)
df_train["transactionRevenue"] = df_train.transactionRevenue.astype(np.float)
y_reg = df_train['transactionRevenue']
features_type = [
    (f,df_train[f].dtype) for f in df_train.columns
]
features_type
df_train["isMobile"] = df_train.isMobile.astype('str')
df_test["isMobile"] = df_test.isMobile.astype('str')
df_train.head()
categorical_features = [
    f for f in df_train.columns if (df_train[f].dtype == 'object') & (f != "fullVisitorId")
]
categorical_features
for f in categorical_features:
    df_train[f], indexer = pd.factorize(df_train[f])
    df_test[f] = indexer.get_indexer(df_test[f])
df_train.shape, df_test.shape
df_train_rf = df_train.copy()
a_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=98.7),
b_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=98.8),
c_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=98.9),
d_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=99),
e_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=99.1),
f_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=99.3),
g_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=99.5),
h_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=99.6),
i_class = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=99.9),
max_revenue = np.percentile(a=list(df_train_rf["transactionRevenue"]) ,q=100),

"""percentiles are: 98.7% percentile: {}, 98.8% percentile: {}, 98.9% percentile: {}, 99.1% percentile: {}, 99.3% percentile: {}, 60% percentile: {},
99.5% percentile: {}, 99.6% percentile: {}, 99.9% percentile: {}, 100% percentile: {}""".format(
    np.percentile(a=a_class ,q=98.7),
    np.percentile(a=b_class ,q=98.8),
    np.percentile(a=c_class ,q=98.9),
    np.percentile(a=d_class ,q=99),
    np.percentile(a=e_class ,q=99.1),
    np.percentile(a=f_class ,q=99.3),
    np.percentile(a=g_class ,q=99.5),
    np.percentile(a=h_class ,q=99.6),
    np.percentile(a=i_class ,q=99.9),
    np.percentile(a=max_revenue ,q=100),
)

df_train_rf.loc[(df_train_rf["transactionRevenue"]<a_class) , "rate"] = 1
df_train_rf.loc[(df_train_rf["transactionRevenue"]<b_class) & (df_train_rf["transactionRevenue"]>=a_class) , "rate"] = 2
df_train_rf.loc[(df_train_rf["transactionRevenue"]<c_class) & (df_train_rf["transactionRevenue"]>=b_class) , "rate"] = 3
df_train_rf.loc[(df_train_rf["transactionRevenue"]<d_class) & (df_train_rf["transactionRevenue"]>=c_class) , "rate"] = 4
df_train_rf.loc[(df_train_rf["transactionRevenue"]<e_class) & (df_train_rf["transactionRevenue"]>=d_class) , "rate"] = 5
df_train_rf.loc[(df_train_rf["transactionRevenue"]<f_class) & (df_train_rf["transactionRevenue"]>=e_class) , "rate"] = 6
df_train_rf.loc[(df_train_rf["transactionRevenue"]<g_class) & (df_train_rf["transactionRevenue"]>=f_class) , "rate"] = 7
df_train_rf.loc[(df_train_rf["transactionRevenue"]<h_class) & (df_train_rf["transactionRevenue"]>=g_class) , "rate"] = 8
df_train_rf.loc[(df_train_rf["transactionRevenue"]<i_class) & (df_train_rf["transactionRevenue"]>=h_class) , "rate"] = 9
df_train_rf.loc[(df_train_rf["transactionRevenue"]>=i_class) , "rate"] = 10


df_train_rf["rate"].astype(int)

df_train_rf = df_train_rf.drop(["transactionRevenue"],axis=1)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=0)
train, test = train_test_split(df_train_rf, test_size=0.2)
rf.fit(train.loc[:, train.columns != "rate"], train[["rate"]])
feature_importance_rev = dict(sorted(zip(map(lambda x: round(x, len(train.columns)), rf.feature_importances_), train.columns),
                                     reverse=True))
feature_importance = dict()
for i, j in feature_importance_rev.items():
    feature_importance[j] = i
feature_importance
import matplotlib.pyplot as plt
%matplotlib inline

feature_importance_df = pd.DataFrame.from_dict(feature_importance,orient="index")
feature_importance_df = feature_importance_df.rename(index=str, columns={0: "importance"})
ax = feature_importance_df.plot(kind='bar', figsize = (12,8))
plt.bar(feature_importance_df.index,feature_importance_df["importance"])
plt.show()
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
df_train.fullVisitorId = df_train["fullVisitorId"].astype('str')
folds = get_folds(df=df_train, n_splits=5)
train_features = [f for f in df_train.columns if f not in  ["fullVisitorId","transactionRevenue"]]
train_features
importances = pd.DataFrame()
oof_reg_preds = np.zeros(df_train.shape[0])
sub_reg_preds = np.zeros(df_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = df_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = df_train[train_features].iloc[val_], y_reg.iloc[val_]
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(X=trn_x,y=np.log1p(trn_y.astype(np.float)),
        eval_set=[(val_x, np.log1p(val_y.astype(np.float)))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(df_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg.astype(np.float)), oof_reg_preds) ** .5
df_test['PredictedLogRevenue'] = np.log1p(sub_reg_preds)
pre_submission = df_test[["fullVisitorId" , "PredictedLogRevenue"]].groupby(by=["fullVisitorId"]).mean()
pre_submission.to_csv(path_or_buf="regression_final.csv", index=True)