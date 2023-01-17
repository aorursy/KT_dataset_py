# Dependencies
from time import time
from dateutil import parser
from pandas.tseries.offsets import BDay
from itertools import chain

import numpy as np
import pandas as pd
import random as rnd
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline
np.random.seed(42)

# Metrics AND FUNCTIONS
# Standardize the data:
from sklearn.preprocessing import StandardScaler

## CLASSIFIERS LIST
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

# Features
f = {'firstMentionSentence':['median','std'],
     'sentimentNeutral':['mean','std'], 
     'noveltyCount12H':['sum'],'noveltyCount24H':['sum'],'noveltyCount3D':['sum'],
     'noveltyCount5D':['sum'],'noveltyCount7D':['sum'],
     'relevance':['median'],  
     'companyCount':['median'], 
     'sentimentNegative':['std'],
     'sentimentWordCount':['median']}
        
def pre_processing(mkt, nws):
    
## CONSOLIDATE TIME TO THE NEXT BUSINESS DAY

    if mkt.time.dtype != 'datetime64[ns, UTC]':
        mkt.time = mkt.time.apply(lambda x: parser.parse(x))
        nws.time = nws.time.apply(lambda x: parser.parse(x))
    nws['time'] = (nws['time'] - np.timedelta64(22,'h')).dt.ceil('1D') #.dt.date 
    mkt['time'] = mkt['time'].dt.floor('1D')
    # Verify if business day, if not, roll to the next B day
    offset = BDay()
    nws.time = nws.time.apply(lambda x: offset.rollforward(x))
    
    ## TRIM
    
    mkt.drop(['assetName','open','returnsClosePrevRaw1','returnsOpenPrevRaw1',
    'returnsClosePrevRaw10','returnsOpenPrevRaw10', 
    'returnsClosePrevMktres1'], axis=1, inplace=True)

    nws.drop(['sourceTimestamp', 'firstCreated', 'sourceId', 'headline',
    'takeSequence', 'provider', 'subjects', 'audiences','bodySize',
    'headlineTag', 'marketCommentary', 'assetName',
    'urgency', 'sentenceCount', 'wordCount', 'sentimentClass', 'sentimentPositive', 'volumeCounts12H',
    'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
    'volumeCounts7D'], axis=1, inplace=True)
   
    ## Break down the set of assets (10s)
    
    nws = expand_assets(nws)

    ## FEATURE Engineering
    
    nws = nws.groupby(['time','assetCode']).agg(f)
    # Correct the labels
    col_name = ['_'.join(title) if isinstance(title, tuple) else title for title in nws.columns ]
    nws.columns = col_name
    # Regroup the noveltyCount variables into max, min, median and std. 
    current_col = [col for col in filter(lambda x: x.startswith('noveltyCount'), nws.columns)]
    nws['novelty_median'] = nws[current_col].apply(np.median, axis=1)
    nws['novelty_std'] = nws[current_col].apply(np.std, axis=1)
    nws.drop(current_col, axis=1, inplace=True)
    
    # MERGING
    
    data = pd.merge(mkt, nws,  how='outer', left_on=['time','assetCode'], right_on = ['time','assetCode'])
    del nws, mkt

    # Set all Nans to 0
    data = data.loc[(~data.volume.isnull()) & (~data.firstMentionSentence_median.isnull())].fillna(0)
    return data

def expand_assets(nws):
    
    nws['assetCodes'] = nws['assetCodes'].str.findall(f"'([\w\./]+)'")   
    assetCodes_expanded = list(chain(*nws['assetCodes']))
    assetCodes_index = nws.index.repeat( nws['assetCodes'].apply(len) )
    df = pd.DataFrame({'idx': assetCodes_index, 'assetCode': assetCodes_expanded})
    # Create expandaded news (will repeat every assetCodes' row)
    nws_expanded = pd.merge(df, nws, left_on='idx', right_index=True)
    nws_expanded.drop(['idx','assetCodes'], axis=1, inplace=True)
    return nws_expanded

def split_dataset(features, data):
    
    # Standardize
    X = StandardScaler().fit_transform(data.loc[:, features].values)
    y = np.array(data.target.values).reshape(X.shape[0],1)
    training_size = np.floor(X.shape[0]*0.75).astype(int)
    X_train = X[:training_size]
    y_train = y[:training_size]
    X_test = X[training_size:]
    y_test = y[training_size:]
    # Many training algorithms are sensitive to the order of the training instances, 
    # so it's generally good practice to shuffle them first:
    rnd_idx = np.random.permutation(training_size)
    X_train = X_train[rnd_idx]
    y_train = y_train[rnd_idx]
    return X_train, X_test, y_train, y_test

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def training(X_train,y_train):

    # --- SGD
    sgd_paste_clf = BaggingClassifier(
        SGDClassifier(random_state=42,alpha=0.01,l1_ratio=0.25,loss='log',penalty='elasticnet'),
        n_estimators=50, n_jobs=1, random_state=40)
    
    # --- DT
    dt_paste_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42,max_leaf_nodes=91,min_samples_leaf=7,min_weight_fraction_leaf=0.01),
        n_estimators=50, n_jobs=1, random_state=40)
    
    # --- Gradient Boosting
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=35, random_state=42)

    sgd_paste_clf.fit(X_train, y_train)
    dt_paste_clf.fit(X_train, y_train)
    gbrt_best.fit(X_train, y_train)

    return sgd_paste_clf, dt_paste_clf, gbrt_best

def get_prediction(clf1, clf2, clf3, X_test):
    yhat = {
    'SGD': clf1.predict_proba(X_test),
    'DT': clf2.predict_proba(X_test)} 
    yhat_GB = {'GB':clf3.predict(X_test)}

    # Hard Voting with Soft Voting output
    my_dict = pd.DataFrame({'clf': list(yhat.keys()),    # score = range(len(tst.valuesof sgd))
            'tags': list(yhat.values())}, columns = ['clf', 'tags'])
    tags = my_dict['tags'].apply(pd.DataFrame)
    df_temp = pd.DataFrame()
    for i,d in enumerate(yhat.keys()):
        tags[i].columns = [d+'-1',d+'+1']
        df_temp = pd.concat([df_temp, tags[i]], axis=1)
    maxCol=lambda x: max(x.min(), x.max(), key=abs)

    df_temp['SGD-1'] = df_temp['SGD-1'].apply(lambda y: y*(-1))
    df_temp['SGD_Label'] = df_temp.loc[:,df_temp.columns.str.startswith('SGD')].apply(maxCol,axis=1)
    df_temp['DT-1'] = df_temp['DT-1'].apply(lambda y: y*(-1))
    df_temp['DT_Label'] = df_temp.loc[:,df_temp.columns.str.startswith('DT')].apply(maxCol,axis=1)

    df_gb = pd.DataFrame(yhat_GB['GB'], columns=['GB'])
    df_gb['GB-1'] = df_gb.GB.apply(lambda x: x-1 if x<=50 else x)
    df_gb.columns = ['GB+1','GB-1']
    df_gb['GB_Label'] = df_gb.loc[:,df_gb.columns.str.startswith('GB')].apply(maxCol,axis=1)

    df_temp = pd.concat([df_temp,df_gb], axis=1)
    df_temp['myHV'] = df_temp.loc[:,df_temp.columns.str.endswith('_Label')].apply(lambda l: np.bincount(l>=0).argmax(), axis=1)

    return df_temp.apply(hVote, axis=1)

def hVote(r):
    avg = []
    for l in ['SGD_Label','DT_Label','GB_Label']:
        if (r.myHV==0 and r[l]<0) or (r.myHV==1 and r[l]>0):
            avg.append(r[l])
    return np.mean(avg)

def make_predictions(predictions_template_df, market_obs_df, news_obs_df):
    # Preprocessing
    df = pre_processing(market_obs_df, news_obs_df)
    df.reset_index(inplace=True)
    df.drop(['time','index'], axis=1, inplace=True)
    features = df.columns.difference(['assetCode'])
        # On the iteration 386 there is so few news that when merged with the market the dataframe has no data. 
    if df.shape[0] > 0:
        X_test = StandardScaler().fit_transform(df.loc[:, features].values)
        # Prediction
#         y_pred = model.predict_proba(X_test)
#         df['confidenceValue'] = [p if p>=0.5 else 1-p for p in y_pred[:,1]]
        df['confidenceValue'] = get_prediction(clf1, clf2, clf3, X_test)
        # Merge prediction with the respective asset code
        pred = pd.merge(predictions_template_df,df.loc[:,['assetCode','confidenceValue']],on='assetCode', how='outer').loc[:,["assetCode", "confidenceValue_y"]].fillna(0)
    else: 
        pred = predictions_template_df.copy()
        pred.columns = ['assetCode','confidenceValue_y']
    predictions_template_df.confidenceValue = pred.confidenceValue_y
    
def get_prediction_2(clf, X_test):
    return clf.predict(X_test)
# First let's import the module and create an environment.
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
# -------- ~ 700 s
start = time()

mkt = reduce_mem_usage(market_train_df)
nws = reduce_mem_usage(news_train_df)
del market_train_df, news_train_df

mkt["returnsOpenNextMktres10"] = mkt["returnsOpenNextMktres10"] > 0 # .clip(-1, 1)
mkt.rename(columns={'returnsOpenNextMktres10':'target'}, inplace=True)

data = pre_processing(mkt, nws)

# Reset Index
data.reset_index(inplace=True)
data.drop(['time','assetCode','index','universe'], axis=1, inplace=True)

# RESULTS
print('Preprocessing Completed:',time()-start, 'seconds')
data.head()
## ---------  ~ 1000s
start = time()

## -------- Split dataset
features = data.columns.difference(['target'])
X_train, X_test, y_train, y_test = split_dataset(features, data)

clf1, clf2, clf3 = training(X_train, y_train)

print(time()-start)
# --- Analytics

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier 

# Results
res = pd.DataFrame()

# Results
def get_res(res, clf_time, clf_name, X_train, y_test, y_pred, p):
    return pd.concat([res,
               pd.DataFrame({'data_size':str(X_train.shape),
              'ETA': clf_time,
              'Acc': accuracy_score(y_test, y_pred),
              'Precision': precision_score(y_test, y_pred),
              'Recall': recall_score(y_test, y_pred),
              'F1': f1_score(y_test, y_pred),
              'MSE': mean_squared_error(y_test, y_pred*1),
              'AUC': roc_auc_score(y_test, y_pred), 
              'Params':p}, index=[clf_name])])

# Function used in feature importance selection
def prep_res(r):
    # Remove the ETA column as well as the Params and data size
    r.drop(['ETA','Params','data_size'], inplace=True, axis=1)
    r = r.stack().reset_index()
    # Join the two indexes together and convert it to a df
    # Merge 
    r['metric'] = r.apply(lambda row: row.level_0+' '+row.level_1, axis=1)
    #res.drop(['level_0','level_1'], inplace=True, axis=1)
    #res.set_index('metric',inplace=True)
    r.columns = ['clf','metric','all','mix']
    return r

def get_baseline(res, X_train, y_train, X_test, y_test, title):
    ## BASELINE  
    # Logistic Regression  95s
    start = time()
    log_clf = LogisticRegression(random_state=42)
    log_clf.fit(X_train, y_train)
    y_pred = log_clf.predict(X_test)
    res = get_res(res, time()-start, 'LogReg', X_train, y_test, y_pred, title)

    # SGD
    start = time()
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    y_pred = sgd_clf.predict(X_test)
    res = get_res(res, time()-start, 'SGD', X_train, y_test, y_pred, title)

    # Decision Tree
    start = time()
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X_train, y_train)
    # tree_clf.predict_proba(X_test)
    tree_clf.predict(X_test)
    res = get_res(res, time()-start, 'DecTree', X_train, y_test, y_pred, title)
    return res
## --------- Results for Analysis
start = time()
res = get_res(pd.DataFrame(), time()-start, 'SGD', X_train, y_test, clf1.predict(X_test), 'Final')
start = time()
res = get_res(res, time()-start, 'DT', X_train, y_test, clf2.predict(X_test), 'Final')
start = time()
res = get_res(res, time()-start, 'GBoost', X_train, y_test, clf3.predict(X_test)>=0.5, 'Final')
start = time()
res = get_res(res, time()-start, 'HVoting', X_train, y_test, get_prediction(clf1, clf2, clf3, X_test)>0, 'Final')

res.loc[:,res.columns.difference(['Params','data_size'])].head()
## ------ Train whole dataset
# ~ 1000 s

X_train = np.vstack((X_train,X_test))
y_train = np.vstack((y_train, y_test))

clf1, clf2, clf3 = training(X_train, y_train)
def make_predictions_2(predictions_template_df, market_obs_df, news_obs_df):
    # Preprocessing
    df = pre_processing(market_obs_df, news_obs_df)
    df.reset_index(inplace=True)
    df.drop(['time','index'], axis=1, inplace=True)
    features = df.columns.difference(['assetCode'])
        # On the iteration 386 there is so few news that when merged with the market the dataframe has no data. 
    if df.shape[0] > 0:
        X_test = StandardScaler().fit_transform(df.loc[:, features].values)
        # Prediction
#         y_pred = model.predict_proba(X_test)
#         df['confidenceValue'] = [p if p>=0.5 else 1-p for p in y_pred[:,1]]
        df['confidenceValue'] = get_prediction_2(clf3, X_test)
        # Merge prediction with the respective asset code
        pred = pd.merge(predictions_template_df,df.loc[:,['assetCode','confidenceValue']],on='assetCode', how='outer').loc[:,["assetCode", "confidenceValue_y"]].fillna(0)
    else: 
        pred = predictions_template_df.copy()
        pred.columns = ['assetCode','confidenceValue_y']
    predictions_template_df.confidenceValue = pred.confidenceValue_y
## ------ Test
# ~ 510 s

start = time()

days = env.get_prediction_days()

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_predictions(predictions_template_df, market_obs_df, news_obs_df)
    env.predict(predictions_template_df)

env.write_submission_file()

print('Done',time()-start)

