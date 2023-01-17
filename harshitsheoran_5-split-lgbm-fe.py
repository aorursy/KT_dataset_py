import numpy as np
import pandas as pd
import random
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import lightgbm as lgb
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train, news_train) = env.get_training_data()
for col in market_train.columns:
    if (market_train[col].dtype == "int64" or market_train[col].dtype == "float64"):
        market_train[col] = market_train[col].fillna(market_train[col].mean())
market_train.head()
market_train['time'] = market_train.time.dt.date
market_train = market_train.loc[market_train['time'] >= date(2010, 1, 1)]
asset_code_dict = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
columns_news = ['firstCreated','relevance','sentimentClass','sentimentNegative','sentimentNeutral',
               'sentimentPositive','noveltyCount24H','noveltyCount7D','volumeCounts24H','volumeCounts7D','assetCodes','sourceTimestamp',
               'assetName','audiences', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence','time']
def feature_engineering(df):
    df['date'] = df['time']
    df['price_diff'] = df['close'] - df['open']
    df['close_to_open'] =  np.abs(df['close'] / df['open'])
    df['assetName_mean_open'] = df.groupby('assetName')['open'].transform('mean')
    df['assetName_mean_close'] = df.groupby('assetName')['close'].transform('mean')
#    market_df.drop(['time'], axis=1, inplace=True)
    
#    news_df = news_df[columns_news]
#    news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour
#    news_df['firstCreated'] = news_df.firstCreated.dt.date
#    news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
#    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
#    news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
#    news_df['len_audiences'] = news_train['audiences'].map(lambda x: len(eval(x)))
#    kcol = ['firstCreated', 'assetCodes']
#    news_df = news_df.groupby(kcol, as_index=False).mean()
#    market_df = pd.merge(market_df, news_df, how='left', left_on=['date', 'assetCode'], 
#                            right_on=['firstCreated', 'assetCodes'])
#    del news_df
    df['assetCodeT'] = df['assetCode'].map(asset_code_dict)

    return df

df = feature_engineering(market_train)
def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == np.float64 or df[col].dtype == np.float32:
            df[col] = df[col].astype(np.float16)
    return df

df = reduce_mem_usage(df)
for col in df.columns:
    if df[col].dtype == "int64" or df[col].dtype == "float16":
        df[col] = df[col].fillna(df[col].mean())
df.head()
df.info()
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from datetime import datetime, date

up = df['returnsOpenNextMktres10'] >= 0

universe = df['universe'].values
d = df['date']

fcol = [c for c in df if c not in ['date', 'assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
print("Creating X")
X = df[fcol].values
print("Creating up")
up = up.values
print("Creating r")
r = df.returnsOpenNextMktres10.values

# Scaling of X values
# It is good to keep these scaling values for later

#f = df[fcol]
#print(f.info())

mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
print("Reshaping X")
X = 1 - ((maxs - X) / rng)
print("Rehaped and asseting")
# Sanity check
assert X.shape[0] == up.shape[0] == r.shape[0]

from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time

# X_train, X_test, up_train, up_test, r_train, r_test,u_train,u_test,d_train,d_test = model_selection.train_test_split(X, up, r,universe,d, test_size=0.25, random_state=99)

df['time'] = pd.to_datetime(df['date'])

te = df['time']>date(2015, 1, 1)

print("Getting index values")

tt = 0
for tt,i in enumerate(te.values):
    if i:
        idx = tt
        print(i,tt)
        break
print(idx)
# for ind_tr, ind_te in tscv.split(X):
#     print(ind_tr)

print("Creating X_train, X_test")

X_train, X_test = X[:idx],X[idx:]

print("Creating up_train, up_test")
up_train, up_test = up[:idx],up[idx:]
print("Creating r_train, r_test")
r_train, r_test = r[:idx],r[idx:]
print("Creating u_train, u_test")
u_train,u_test = universe[:idx],universe[idx:]
print("Creating d_train, d_test")
d_train,d_test = d[:idx],d[idx:]
print("Creating Train Dataset")
train_data = lgb.Dataset(X_train, label=up_train.astype(int))
# train_data = lgb.Dataset(X, label=up.astype(int))
print("Creating Test Dataset")
test_data = lgb.Dataset(X_test, label=up_test.astype(int))
from sklearn.model_selection import StratifiedKFold
DATA_SPLIT_SEED = 2018

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=DATA_SPLIT_SEED).split(X, up))
for idx, (train_idx, valid_idx) in enumerate(splits):
    X_train = X[train_idx]
    y_train = up[train_idx]
    X_val = X[valid_idx]
    y_val = up[valid_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_val, label=y_val)
    
    x_1 = [0.19000424246380565, 2452, 212, 328, 202]
    x_2 = [0.19016805202090095, 2583, 213, 312, 220]

    def exp_loss(p,y):
        y = y.get_label()
#        p = p.get_label()
        grad = -y*(1.0-1.0/(1.0+np.exp(-y*p)))
        hess = -(np.exp(y*p)*(y*p-1)-1)/((np.exp(y*p)+1)**2)
    
        return grad,hess

    params_1 = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
#            'objective': 'regression',
            'learning_rate': x_1[0],
            'num_leaves': x_1[1],
            'min_data_in_leaf': x_1[2],
#            'num_iteration': x_1[3],
            'num_iteration': 239,
            'max_bin': x_1[4],
            'verbose': 1
    }

    params_2 = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
#            'objective': 'regression',
            'learning_rate': x_2[0],
            'num_leaves': x_2[1],
            'min_data_in_leaf': x_2[2],
#            'num_iteration': x_2[3],
            'num_iteration': 172,
            'max_bin': x_2[4],
            'verbose': 1
    }

    gbm_1 = lgb.train(params_1,
            train_data,
            num_boost_round=100,
            valid_sets=test_data,
            early_stopping_rounds=5,
#            fobj=exp_loss,
        )

    gbm_2 = lgb.train(params_2,
            train_data,
            num_boost_round=100,
            valid_sets=test_data,
            early_stopping_rounds=5,
#            fobj=exp_loss,
        )
confidence_test = (gbm_1.predict(X_test) + gbm_2.predict(X_test))/2
confidence_test = (confidence_test-confidence_test.min())/(confidence_test.max()-confidence_test.min())
confidence_test = confidence_test*2-1
print(max(confidence_test),min(confidence_test))

# calculation of actual metric that is used to calculate final score
r_test = r_test.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_test * r_test * u_test
data = {'day' : d_test, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_test = mean / std
print(score_test)
import matplotlib.pyplot as plt
import seaborn as sns
feat_importance = pd.DataFrame()
feat_importance["feature"] = fcol
feat_importance["gain"] = gbm_1.feature_importance(importance_type='gain')
feat_importance.sort_values(by='gain', ascending=False, inplace=True)
plt.figure(figsize=(8,10))
ax = sns.barplot(y="feature", x="gain", data=feat_importance)
plt.show()
days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
def post_scaling(df):
    mean, std = np.mean(df), np.std(df)
    df = (df - mean)/ (std * 8)
    return np.clip(df,-1,1)
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if (n_days%50==0):
        print(n_days,end=' ')
    for col in market_obs_df.columns:
        if (market_obs_df[col].dtype == "int64" or market_obs_df[col].dtype == "float64"):
            market_obs_df[col] = market_obs_df[col].fillna(market_obs_df[col].mean())
    
    for col in market_obs_df.columns:
        if (market_obs_df[col].dtype == "int64" or market_obs_df[col].dtype == "float64"):
            market_obs_df[col] = market_obs_df[col].fillna(market_obs_df[col].mean())
            
    market_obs_df = feature_engineering(market_obs_df)
    
    fcol = [c for c in market_obs_df if c not in ['date', 'assetCode', 'time', 'returnsOpenNextMktres10', 'universe', 'assetName', 'Unnamed: 0']]
    
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_live = market_obs_df[fcol].values
#    X_live = 1 - ((maxs - X_live) / rng)
    
    lp = (gbm_1.predict(X_live) + gbm_2.predict(X_live))/2
    

    confidence = lp
#    confidence = (confidence-confidence.min())/(confidence.max()-confidence.min())
    confidence = confidence * 2 - 1
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence': post_scaling(confidence)})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
env.write_submission_file()
sub  = pd.read_csv("submission.csv")
sub.head()

