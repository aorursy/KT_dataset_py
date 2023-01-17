import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn import preprocessing


columnPredict = ['channelGrouping', 'date', 'visitId', 'visitNumber', 'visitStartTime', 'browser', 'deviceCategory', 'isMobile', 'operatingSystem', 'city', 'continent', 'country', 'metro', 'networkDomain', 'region', 'subContinent', 'bounces', 'hits', 'newVisits', 'pageviews', 'adContent', 'campaign', 'isTrueDirect', 'keyword', 'medium', 'source']
def saveData(train,test):
    train.to_csv('trainCategorical.csv',index=False)
    test.to_csv('testCategorical.csv',index=False)
def loadData():
    train = pd.read_csv('trainCategorical.csv',low_memory=False)
    test = pd.read_csv('testCategorical.csv',low_memory=False)
    return train,test


def load_df(csv_path='train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv('../input/'+csv_path, dtype={'fullVisitorId': 'str'})

    for column in JSON_COLUMNS:
        df = df.join(pd.DataFrame(df.pop(column).apply(pd.io.json.loads).values.tolist(), index=df.index))

    return df

def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.values)
    result = pd.DataFrame(x_scaled)
    return result
# parsing data

def parsingDt(fileName):
    data = load_df(csv_path = fileName)
    data = data.drop(columns=['adwordsClickInfo','referralPath'])
    const_cols = [c for c in data.columns if data[c].nunique(dropna=False)==1 ]
    data = data.drop(columns=const_cols)
    return data
    #data.to_csv(str(fileName) +'_f1.csv')

train = parsingDt('train.csv')
train = train.drop(columns=['campaignCode'])
print('TRAIN FINISH')
test = parsingDt('test.csv')
print('TEST FINISH')


print(list(train))
print("Variables not in test but in train : ", set(train.columns).difference(set(test.columns)))
saveData(train,test)
# data conversion

from sklearn import model_selection, preprocessing, metrics
train,test = loadData()
train["transactionRevenue"].fillna(0, inplace=True)
cat_cols = ["channelGrouping", "browser", 
            "deviceCategory", "operatingSystem", 
            "city", "continent", 
            "country", "metro",
            "networkDomain", "region", 
            "subContinent", "adContent", "campaign",
            "keyword", "medium", 
            "source",'isTrueDirect']
for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

num_cols = ['channelGrouping','visitId','visitNumber','browser','deviceCategory','isMobile','hits','newVisits','pageviews','adContent','campaign','isTrueDirect','keyword','medium','source','visitStartTime','bounces','operatingSystem','city','continent','country','networkDomain','region','subContinent',]
for col in num_cols:
    train[col] = train[col].astype(float)
    test[col] = test[col].astype(float)

saveData(train,test)
import numpy as np
import pandas as pd
import datetime
train,test = loadData()


train_id = train["fullVisitorId"].values
test_id = test["fullVisitorId"].values

dev_df = train[train['date']<=int(datetime.date(2017,5,30).strftime('%Y%m%d'))]
val_df = train[train['date']>int(datetime.date(2017,5,30).strftime('%Y%m%d'))]
Ytrain = np.log1p(dev_df["transactionRevenue"].values)
Ytest = np.log1p(val_df["transactionRevenue"].values)

Xtrain = dev_df[columnPredict]
Xtest = val_df[columnPredict]


predictX = test[columnPredict]

#regression models
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

xgb_params = {
        'eval_metric':'rmse',
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 21,
        'min_child_weight': 53,
        'gamma' : 1.31,
        'alpha': 0.0,
        'lambda': 0.5,
        'subsample': 0.63,
        'colsample_bytree': 0.051,
        'colsample_bylevel': 0.51,
        'n_jobs': -1,
        'random_state': 345
    }

        
lgb_params = {
    'learning_rate': 0.03,
    'metric': 'rmse',
    "num_leaves" : 30,
    "min_child_samples" : 90,
    "learning_rate" : 0.02,
    "bagging_fraction" : 0.9,
    "feature_fraction" : 0.6,
    "bagging_frequency" : 5,
    "bagging_seed" : 2018,
    "verbosity" : -1
}

def sv(Regressor,resVal):
    X_tr = np.dstack((Regressor['lgb'],Regressor['xgb'])).reshape((-1,2))
    X_pr =np.dstack((resVal['lgb'],resVal['xgb'])).reshape((-1,2))
    print(X_tr.shape)
    print(X_pr.shape)
    print(len(Regressor['tValue']))
    reg = LinearRegression().fit(X_tr,Regressor['tValue'])
    print(reg.score(X_tr, Regressor['tValue']))
    return reg.predict(X_pr)

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    models = {'lgb': LGBMRegressor(**lgb_params, n_estimators = 500), 
              'xgb': XGBRegressor(**xgb_params, n_estimators = 400)}
    
    ResRegressor = {'tValue': np.array([]),
                    'lgb': np.array([]), 
             'xgb': np.array([])}
    
    Yresult = {'lgb': np.array([]), 
             'xgb': np.array([])}
    
    for name in ['xgb', 'lgb']:
        models[name].fit(train_X, train_y, eval_set = [(val_X, val_y)],early_stopping_rounds = 100, eval_metric = 'rmse', verbose = False)
        val_pred = models[name].predict(val_X)
        val_pred[val_pred<0] = 0
        ResRegressor[name] = val_pred
        print(np.sqrt(mean_squared_error(val_pred, val_y)))
        result_pred = models[name].predict(test_X)
        result_pred[result_pred < 0] = 0
        Yresult[name] = result_pred
    ResRegressor['tValue'] = val_y
    return Yresult,ResRegressor
        
        
Yres,Regressor = run_lgb(Xtrain, Ytrain,Xtest, Ytest, predictX)
pred_test = sv(Regressor,Yres)
sub_df = pd.DataFrame({"fullVisitorId":test_id})
sample_submission = pd.read_csv('../input/sample_submission.csv')
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])

result = pd.merge(sample_submission[['fullVisitorId']],sub_df, left_on='fullVisitorId', right_on='fullVisitorId', how='left')
result['PredictedLogRevenue'] = result['PredictedLogRevenue'].fillna(0)
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
result.to_csv("LG.csv", index=False)

