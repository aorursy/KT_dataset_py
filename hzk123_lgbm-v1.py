import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(marketdf, newsdf) = env.get_training_data()

#back_market = marketdf.copy()
from multiprocessing import Pool
return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close' ]
add_features=['pricevolume' , 'bartrend' , 'average' , 'returnsOpenNextMktres10']
def create_lag(df_code,n_lag=[3,7,10], shift_size=1):
    code = df_code['assetCode'].unique()
    for col in return_features:
        for window in n_lag:
            rolled = df_code[col].shift(shift_size).rolling(window=window)
            lag_mean = rolled.mean()
            df_code['%s_lag_%s_mean'%(col,window)] = lag_mean
    return df_code.fillna(-1)

def generate_lag_features(df,n_lag = [3,7,10]):
    features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10', 'universe']
    
    assetCodes = df['assetCode'].unique()
    #print(assetCodes)
    all_df = []
    
    df['pricevolume'] = df['volume'] * df['close']
    df['bartrend'] = df['close'] / df['open']
    df['average'] = (df['close'] + df['open'])/2
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode'] + add_features +  return_features ] for df_code in df_codes]
    #print('total %s df'%len(df_codes))
    
    pool = Pool(4)
    all_df = pool.map(create_lag, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(return_features,axis=1,inplace=True)
    pool.close()
    
    return new_df
#n_lag = [1,3,5,7,9]
#new_marketdf = generate_lag_features(back_market.head(500),  n_lag)
#print(new_marketdf.columns)
#return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']#
#n_lag = [3,7,14]
#new_df = generate_lag_features(market_train_df,n_lag=n_lag)
#market_train_df = pd.merge(market_train_df,new_df,how='left',on=['time','assetCode'])
def prepare_data(marketdf, newsdf ):
    # a bit of feature engineering
    
    n_lag = [3,7,10]
    marketdf = generate_lag_features(marketdf, n_lag = n_lag)
    
    marketdf['time'] = marketdf.time.dt.strftime("%Y%m%d").astype(int)

    
    
    newsdf['time'] = newsdf.time.dt.strftime("%Y%m%d").astype(int)
    newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])
    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']
    newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']
    

    # get rid of extra junk from news data
    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'assetCodes','urgency','wordCount','sentimentWordCount']
    newsdf.drop(droplist, axis=1, inplace=True)
    #marketdf.drop(['assetName', 'volume'], axis=1, inplace=True)
    
    # combine multiple news reports for same assets on same day
    newsgp = newsdf.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    
    # join news reports to market data, note many assets will have many days without news data
    return pd.merge(marketdf, newsgp, how='left', on=['time', 'assetCode'], copy=False) #, right_on=['time', 'assetCodes'])

cdf = prepare_data(marketdf, newsdf)    
#pd.set_option('display.max_columns', None)
#显示所有行
#pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
#pd.set_option('max_colwidth',100)

#corr = cdf.corr();
#print(corr['sentimentPositive'])

#print(cdf.corr()['sentimentPositive'])

print('building training set...')
targetcols = ['returnsOpenNextMktres10']
traincols = [col for col in cdf.columns if col not in ['time', 'assetCode', 'universe'] + targetcols]

dates = cdf['time'].unique()
train = range(len(dates))[:int(0.95*len(dates))]
val = range(len(dates))[int(0.95*len(dates)):]

# we be classifyin
cdf[targetcols[0]] = (cdf[targetcols[0]] > 0).astype(int)


# train data
Xt = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[train])].values
Yt = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[train])].values

# validation data
Xv = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[val])].values
Yv = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[val])].values


#######################################################
##
## LightGBM
##
#######################################################
import lightgbm as lgb
print ('Training lightgbm')

# money
params = {"objective" : "binary",
          "metric" : "binary_logloss",
          "num_leaves" : 60,
          "max_depth": -1,
          "learning_rate" : 0.01,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 2333,
          "verbosity" : -1 }


lgtrain, lgval = lgb.Dataset(Xt, Yt[:,0]), lgb.Dataset(Xv, Yv[:,0])
lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, verbose_eval=200)


preddays = env.get_prediction_days()

print("generating predictions...")

add_features=['pricevolume' , 'bartrend' , 'average' ]
for (marketdf, newsdf, predtemplatedf) in preddays:
    cdf = prepare_data(marketdf, newsdf)
    Xp = cdf[traincols].fillna(0).values
    preds = lgbmodel.predict(Xp, num_iteration=lgbmodel.best_iteration) * 2 - 1
    predsdf = pd.DataFrame({'ast':cdf['assetCode'],'conf':preds})
    predtemplatedf['confidenceValue'][predtemplatedf['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values
    env.predict(predtemplatedf)

env.write_submission_file()

print('done!!!!!!!')