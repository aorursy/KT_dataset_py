
import pandas as pd
import numpy as np
from sklearn import *
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
market_train = market_train[market_train.returnsOpenNextMktres10<1][market_train.returnsOpenNextMktres10>-1]
def calcRsi(series, period = 14):
    
    """
    Calculate the RSI of a data series 
    
    Parameters
    ----------
    series : pandas series
        Candle sticks dataset
    period : int
        Period of each calculation
        
    Returns
    -------
    rsi : float
        the calculated rsi
    """
    try:
        delta = series.diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
        u = u.drop(u.index[:(period-1)])
        d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
        d = d.drop(d.index[:(period-1)])

        rs = u.ewm(com=period-1, adjust=False).mean() \
            / d.ewm(com=period-1, adjust=False).mean()
        
        rsi = 100 - 100 / (1 + rs)
    except IndexError:
        rsi = 0
    
    return rsi

def addBollinger(df, period=20, col='close'):
    """
    Add the simple moving average column to dataframe 

    Parameters
    ----------
    df : pandas dataframe
        Candle sticks dataset
    period : int
        Period of each calculation

    Returns
    -------
    None
    """
    bbmid_series = df[col].rolling(window=period).mean()
    series_stdev = df[col].rolling(window=period).std()
    df['BBUpperBand'] = bbmid_series + 2*series_stdev
    df['BBLowerBand'] = bbmid_series - 2*series_stdev
    df['BBBandwidth'] = df['BBUpperBand'] - df['BBLowerBand']  
    df['BBMiddleBand'] = bbmid_series
    return df

def addMACD(df):
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    signal_line = df['close'].ewm(span=9).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df.macd.ewm(span=9, adjust=False).mean()
    df['macdh'] = df['macd'] - df['macd_signal']
    return df

def allindicator(market_train):
    market_train['RSI'] = calcRsi(market_train['close'], 14)
    market_train = addBollinger(market_train)
    market_train = addMACD(market_train)
    return market_train

def EMA(df):
    df = df.groupby('time')
    EMA_7 = df['close'].ewm(span=7).mean()
    return df

from multiprocessing import Pool

def groupAsset(market_train):
    all_df = []
    df_codes = market_train.groupby('assetCode')
    df_codes = [df_code[1] for df_code in df_codes]
    pool = Pool(4)
    all_df = pool.map(allindicator, df_codes)
    new_df = pd.concat(all_df)  
    pool.close()
    del all_df, df_codes, market_train
    return new_df
def data_prep(market_train,news_train):
    
    #market_train = groupAsset(market_train)
    
    market_train['formatdate']=market_train['time'].dt.date
    news_train['formatdate'] = news_train.firstCreated.dt.date
    
    for col_cat in ['headlineTag', 'provider', 'sourceId']:
        news_train[col_cat], uniques = pd.factorize(news_train[col_cat])
        del uniques
    
    news_train_sim = news_train.drop(columns=['time','sourceTimestamp','firstCreated','headline','subjects','audiences','assetCodes'])
    news_train_1 = news_train_sim.groupby(by=['assetName','formatdate'],as_index=False,observed=True).mean().add_suffix('_mean')
    news_train_1.rename(columns={'assetName_mean': 'assetName', 'formatdate_mean': 'formatdate'}, inplace=True)
    #news_train_2 = news_train_sim.groupby(by=['assetName','formatdate'],as_index=False,observed=True).sum().add_suffix('_sum')
    #news_train_2.rename(columns={'assetName_sum': 'assetName', 'formatdate_sum': 'formatdate'}, inplace=True)
    #news_train_3 = news_train_sim.groupby(by=['assetName','formatdate'],as_index=False,observed=True).var()
    #news_train_group = pd.merge(news_train_1,news_train_2.iloc[:, np.r_[0,1,2:3]],on=['assetName','formatdate'])
    #,news_train_3,on=['assetName','formatdate'])
    merge_train = pd.merge(market_train, news_train_1, how='left', left_on=['formatdate', 'assetName'], 
                                right_on=['formatdate', 'assetName'])
    #merge_train.fillna(0,inplace=True)
    #merge_train.drop(['marketCommentary_sum','noveltyCount24H_mean','noveltyCount3D_mean','marketCommentary_mean','urgency_sum',
    #                 'urgency_mean','noveltyCount5D_mean','bodySize_sum','noveltyCount12H_mean','provider_sum','sentimentClass_mean',
    #                 'takeSequence_sum','volumeCounts12H_mean','headlineTag_mean','wordCount_sum','headlineTag_sum'],1,inplace=True)
    del market_train, news_train_sim, news_train_1
    
    #merge_train.apply(lambda x: x.fillna(x.mean()),axis=0)
    
    
    
    return merge_train


#testabc['change']


merge_train = data_prep(market_train,news_train)

df_main = merge_train[['time','assetCode','close']]
df_main = groupAsset(df_main)
merge_train  = pd.merge(merge_train, df_main, how='left',on=['time','assetCode','close'])
df_main = df_main[['time','assetCode','close']]

test123 = merge_train[['time','close','volume']]
test123['cap']=test123['close']*test123['volume']
testabc = test123.groupby('time').sum()
testabc['wholemarket_index']=testabc['cap']/testabc['volume']
testabc.reset_index(inplace=True)
for i in range (len(testabc)-1):
    testabc.loc[i+1,'changeWMI'] = (testabc.loc[i+1,'wholemarket_index']-testabc.loc[i,'wholemarket_index'])/(testabc.loc[i,'wholemarket_index'])
testabc = testabc[['time','changeWMI']]
merge_train  = pd.merge(merge_train, testabc, how='left',on=['time'])
test123 = merge_train[['time','close','volume']]

merge_train.fillna(0,inplace=True)

up = merge_train.returnsOpenNextMktres10 >= 0
up = up.values

fcol = [c for c in merge_train.columns if c not in ['assetCode','assetName','time',
                                             'returnsOpenNextMktres10', 'formatdate', 'universe','close','open']]
X = merge_train[fcol].values
r = merge_train.returnsOpenNextMktres10.values

# Scaling of X values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
test123
X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.1, random_state=99)

import lightgbm as lgb
params = {'learning_rate': 0.15, 'max_depth': 20, 'num_leaves':512, 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss', 'is_training_metric': True, 'seed': 42}
model = lgb.train(params, train_set=lgb.Dataset(X_train, label=up_train), num_boost_round=100,
                  valid_sets=[lgb.Dataset(X_train, label=up_train), lgb.Dataset(X_test, label=up_test)],
                  verbose_eval=50, early_stopping_rounds=30)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: np.random.randint(0, 255), range(3)))
    return color

df = pd.DataFrame({'imp': model.feature_importance(), 'col':fcol})
df = df.sort_values(['imp','col'], ascending=[True, False])
data = [df]
for dd in data:  
    colors = []
    for i in range(len(dd)):
         colors.append(generate_color())

    data = [
        go.Bar(
        orientation = 'h',
        x=dd.imp,
        y=dd.col,
        name='Features',
        textfont=dict(size=20),
            marker=dict(
            color= colors,
            line=dict(
                color='#000000',
                width=0.5
            ),
            opacity = 0.87
        )
    )
    ]
    layout= go.Layout(
        title= 'Feature Importance of LGB',
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )

    py.iplot(dict(data=data,layout=layout), filename='horizontal-bar')



df_main = df_main[df_main.time>'2016-12-01']
n_days = 0 
for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
    n_days +=1
    if n_days % 2 == 0:
        print(n_days,end=' ')
    market_obs_df = data_prep(market_obs_df, news_obs_df)
    
    df_days = market_obs_df[['time','assetCode','close']]
    df_main = df_main.append(df_days)
    df_main.sort_values(by=['time'],inplace=True)
    df_main.reset_index(inplace=True,drop=True)
    df_main = groupAsset(df_main)
    market_obs_df  = pd.merge(market_obs_df, df_main, how='left',on=['time','assetCode','close'])
    df_main = df_main[['time','assetCode','close']]
    
    test456 = market_obs_df[['time','close','volume']]
    test123 = test123.append(test456)
    test123['cap']=test123['close']*test123['volume']
    testabc = test123.groupby('time').sum()
    testabc['wholemarket_index']=testabc['cap']/testabc['volume']
    testabc.reset_index(inplace=True)
    for i in range (len(testabc)-1):
        testabc.loc[i+1,'changeWMI'] = (testabc.loc[i+1,'wholemarket_index']-testabc.loc[i,'wholemarket_index'])/(testabc.loc[i,'wholemarket_index'])
    testabc = testabc[['time','changeWMI']]
    market_obs_df  = pd.merge(market_obs_df, testabc, how='left',on=['time'])
    test123 = test123[['time','close','volume']]
    
    market_obs_df.fillna(0,inplace=True)
    
    X_live = market_obs_df[fcol].values
    
    # Scaling of X values
    #mins = np.min(X_live, axis=0)
    #maxs = np.max(X_live, axis=0)
    #rng = maxs - mins
    X_live = 1 - ((maxs - X_live) / rng)
    
    
    
    
    lp = model.predict(X_live)
    
    confidence = 2 * lp -1
      
    #market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)


env.write_submission_file()
