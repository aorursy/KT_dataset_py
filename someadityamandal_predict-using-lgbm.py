# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#import
from kaggle.competitions import twosigmanews
from datetime import datetime, date
import numpy as np
from sklearn import model_selection

# Any results you write to the current directory are saved as output.
env = twosigmanews.make_env()
(market_train_orig, news_train_orig) = env.get_training_data()
market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
market_train_df['time'] = market_train_df['time'].dt.date
market_train_df = market_train_df.loc[market_train_df['time']>=date(2009, 1, 1)]
news_train_df['time'] = news_train_df['time'].dt.date
news_train_df = news_train_df.loc[news_train_df['time']>=date(2009, 1, 1)]
del market_train_orig
del news_train_orig
from multiprocessing import Pool

def create_lag(df_code,n_lag=[3,7,14,],shift_size=1):
    code = df_code['assetCode'].unique()
    
    for col in return_features:
        for window in n_lag:
            rolled = df_code[col].shift(shift_size).rolling(window=window)
            lag_mean = rolled.mean()
            lag_max = rolled.max()
            lag_min = rolled.min()
            lag_std = rolled.std()
            df_code['%s_lag_%s_mean'%(col,window)] = lag_mean
            df_code['%s_lag_%s_max'%(col,window)] = lag_max
            df_code['%s_lag_%s_min'%(col,window)] = lag_min
#             df_code['%s_lag_%s_std'%(col,window)] = lag_std
    return df_code.fillna(-1)

def generate_lag_features(df,n_lag = [3,7,14]):
    features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10', 'universe']
    
    assetCodes = df['assetCode'].unique()
    print(assetCodes)
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode']+return_features] for df_code in df_codes]
    print('total %s df'%len(df_codes))
    
    pool = Pool(4)
    all_df = pool.map(create_lag, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(return_features,axis=1,inplace=True)
    pool.close()
    
    return new_df
return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']
n_lag = [3,7,14]
new_df = generate_lag_features(market_train_df,n_lag=n_lag)
market_train_df = pd.merge(market_train_df,new_df,how='left',on=['time','assetCode'])
del new_df
import gc
gc.collect()
print(market_train_df.columns)
def prepare_news_data(news_df):
    news_df['position'] = news_df['firstMentionSentence'] / news_df['sentenceCount']
    news_df['coverage'] = news_df['sentimentWordCount'] / news_df['wordCount']

    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'urgency','wordCount','noveltyCount12H','sentimentWordCount','noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D','volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D',]
    news_df.drop(droplist, axis=1, inplace=True)

    # create a mapping between 'assetCode' to 'news_index'
    assets = []
    indices = []
    for i, values in news_df['assetCodes'].iteritems():
        assetCodes = eval(values)
        assets.extend(assetCodes)
        indices.extend([i]*len(assetCodes))
    mapping_df = pd.DataFrame({'news_index': indices, 'assetCode': assets})
    del assets, indices
    
    # join 'news_train_df' and 'mapping_df' (effectivly duplicating news entries)
    news_df['news_index'] = news_df.index.copy()
    expanded_news_df = mapping_df.merge(news_df, how='left', on='news_index')
    del mapping_df, news_df
    
    expanded_news_df.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return expanded_news_df.groupby(['time', 'assetCode']).mean().reset_index()
news_train_df = prepare_news_data(news_train_df)
market_train_df = market_train_df.merge(news_train_df, how='left', on=['assetCode', 'time'])
market_train_df
del news_train_df
gc.collect
def fill_in(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

market_train_df = fill_in(market_train_df)
def data_other(market_train_df):
    #market_train_df.time = market_train_df.time.dt.date
    lbl = {k: v for v, k in enumerate(market_train_df['assetCode'].unique())}
    market_train_df['assetCodeT'] = market_train_df['assetCode'].map(lbl)
    
    market_train_df = market_train_df.dropna(axis=0)
    
    return market_train_df

market_train_df = data_other(market_train_df)
market_train_df
green = market_train_df.returnsOpenNextMktres10 > 0
green = green.values
fcol = [c for c in market_train_df if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
X = market_train_df[fcol].values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
all_10day = market_train_df.returnsOpenNextMktres10.values
X_train, X_test, green_train, green_test, all_train, all_test = model_selection.train_test_split(
    X, green, all_10day, test_size=0.20, random_state=59)
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=green_train.astype(int))
test_data = lgb.Dataset(X_test, label=green_test.astype(int))
# these are tuned params I found
x_1 = [0.19000424246380565, 2452, 212, 328, 202]
x_2 = [0.19016805202090095, 2583, 213, 312, 220]
params_1 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': x_1[0],
        'num_leaves': x_1[1],
        'min_data_in_leaf': x_1[2],
        'num_iteration': x_1[3],
        'max_bin': x_1[4],
        'verbose': 1
    }

params_2 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': x_2[0],
        'num_leaves': x_2[1],
        'min_data_in_leaf': x_2[2],
        'num_iteration': x_2[3],
        'max_bin': x_2[4],
        'verbose': 1
    }


gbm_1 = lgb.train(params_1,
        train_data,
        num_boost_round=1000,
        valid_sets=test_data,
        early_stopping_rounds=100)
        
gbm_2 = lgb.train(params_2,
        train_data,
        num_boost_round=1000,
        valid_sets=test_data,
        early_stopping_rounds=100)
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
days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
total_market_obs_df = []
fcol1 = [c for c in market_train_df if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp','companyCount','relevance','sentimentNegative','sentimentNeutral','sentimentPositive','position','coverage']]
X = market_train_df[fcol1].values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
import pandas as pd
from sklearn import preprocessing
import time
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if (n_days%50==0):
        print(n_days,end=' ')
    t = time.time()
    market_obs_df['time'] = market_obs_df['time'].dt.date
    
    return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']
    total_market_obs_df.append(market_obs_df)
    if len(total_market_obs_df)==1:
        history_df = total_market_obs_df[0]
    else:
        history_df = pd.concat(total_market_obs_df[-(np.max(n_lag)+1):])
    print(history_df)
    
    new_df = generate_lag_features(history_df,n_lag=[3,7,14])
    market_obs_df = pd.merge(market_obs_df,new_df,how='left',on=['time','assetCode'])
    
    #news_obs_df = prepare_news_data(news_obs_df)
    #market_obs_df = market_obs_df.merge(news_obs_df, how='left', on=['assetCode', 'time'])
    
#     return_features = ['open']
#     new_df = generate_lag_features(market_obs_df,n_lag=[3,7,14])
#     market_obs_df = pd.merge(market_obs_df,new_df,how='left',on=['time','assetCode'])
    
    market_obs_df = fill_in(market_obs_df)
    
    market_obs_df = data_other(market_obs_df)
    
#     market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    
    X_live = market_obs_df[fcol1].values
    X_live = 1 - ((maxs - X_live) / rng)
    prep_time += time.time() - t
    
    t = time.time()
    lp = (gbm_1.predict(X_live) + gbm_2.predict(X_live))/2
    prediction_time += time.time() -t
    
    t = time.time()
    
    confidence = lp
    confidence = (confidence-confidence.min())/(confidence.max()-confidence.min())
    confidence = confidence * 2 - 1
    
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    
#     predictions_template_df.set_index('assetCode')
#     x = predictions_template_df[['confidenceValue']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#     np_scaled = min_max_scaler.fit_transform(x)
#     df_normalized = pd.DataFrame(np_scaled,columns=predictions_template_df[['confidenceValue']])
#     df_normalized['assetCode'] = df_normalized.index
    
    predictions_template_df[['confidenceValue']] = min_max_scaler.fit_transform(predictions_template_df[['confidenceValue']])
    
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    
env.write_submission_file()
sub  = pd.read_csv("submission.csv")
