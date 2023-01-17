import pandas as pd
import numpy as np
from sklearn import *
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
#pd.options.display.max_rows = 200
#market_train[market_train.returnsOpenNextMktres10.abs()>1]

#market_train[market_train.assetCode=="ADI.N"]


#check and drop duplicates
#market_train_sim[market_train_sim[['formatdate','assetCode']].duplicated()]
#news_train_group.drop_duplicates(inplace=True)
#filter
#news_train_sim[news_train_sim.assetName.str.contains('Unknown')]
#market_train[market_train.assetName.str.contains('Caleres Inc')]
#merge_train[merge_train.returnsOpenNextMktres10 > 1]


#news_train[news_train.sourceId.value_counts()>1]
#asset_Time.groupby(['assetName','formatdate']).size()

#merge_train.returnsOpenNextMktres10.quantile(.9999)
# Import the libraries
#import matplotlib.pyplot as plt
#import seaborn as sns

# matplotlib histogram
#plt.hist(merge_train['returnsOpenNextMktres10'], color = 'blue', edgecolor = 'black',
#         bins = 1)

# seaborn histogram
#sns.distplot(merge_train['returnsOpenNextMktres10'], hist=True, kde=False, 
#             bins=1, color = 'blue',
#             hist_kws={'edgecolor':'black'})
# Add labels
#plt.title('Histogram of Arrival Delays')
#plt.xlabel('Delay (min)')
#plt.ylabel('Flights')
#scatterplot
#fig, ax = plt.subplots(figsize=(16,8))
#ax.scatter(market_train['returnsClosePrevMktres10'], market_train['returnsOpenNextMktres10'])
#ax.set_xlabel('returnsClosePrevMktres10')
#ax.set_ylabel('returnsOpenNextMktres10')
#lt.show()
#news_train['time'] = news_train.time.dt.hour
#news_train['asset_sentiment_count'] = news_train.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
#market_train['volume_to_mean'] = market_train['volume'] / market_train[['volume','assetCode']].mean()
market_train = market_train[market_train.returnsOpenNextMktres10<1][market_train.returnsOpenNextMktres10>-1]

#[news_train['assetName']=='PetroChina Co Ltd'][news_train['asset_sentiment_count']==3992]
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
    
    market_train = groupAsset(market_train)
    
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
    merge_train.fillna(0,inplace=True)
    #merge_train.drop(['marketCommentary_sum','noveltyCount24H_mean','noveltyCount3D_mean','marketCommentary_mean','urgency_sum',
    #                 'urgency_mean','noveltyCount5D_mean','bodySize_sum','noveltyCount12H_mean','provider_sum','sentimentClass_mean',
    #                 'takeSequence_sum','volumeCounts12H_mean','headlineTag_mean','wordCount_sum','headlineTag_sum'],1,inplace=True)
    del market_train, news_train_sim, news_train_1
    
    #merge_train.apply(lambda x: x.fillna(x.mean()),axis=0)
    
    
    
    return merge_train





merge_train = data_prep(market_train,news_train)
up = merge_train.returnsOpenNextMktres10 >= 0
up = up.values

fcol = [c for c in merge_train.columns if c not in ['assetCode','assetName','time',
                                             'returnsOpenNextMktres10', 'formatdate', 'universe']]
X = merge_train[fcol].values
r = merge_train.returnsOpenNextMktres10.values

# Scaling of X values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)


X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.1, random_state=99)

import lightgbm as lgb
params = {'learning_rate': 0.25, 'max_depth': 10, 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss', 'is_training_metric': True, 'seed': 42}
model = lgb.train(params, train_set=lgb.Dataset(X_train, label=up_train), num_boost_round=2000,
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
merge_train

n_days = 0 
for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
    n_days +=1
    if n_days % 50 == 0:
        print(n_days,end=' ')
    market_obs_df = data_prep(market_obs_df, news_obs_df)    
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
#groupA = merge_train.groupby('assetCode')['time'].count().reset_index(name = 'count').sort_values('count').head(100)
#groupA
#import plotly.graph_objs as go
#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#data = []
##for asset in np.random.choice(market_train_df['assetName'].unique(), 10):
#asset = "ADI.N"
#asset_df = merge_train[(merge_train['assetCode'] == asset)]

#data.append(go.Scatter(
#    x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
#    y = asset_df['BBUpperBand'].values,
#    name = asset
#))
#layout = go.Layout(dict(title = "Closing prices of 10 random assets",
#              xaxis = dict(title = 'Month'),
#              yaxis = dict(title = 'Price (USD)'),
#              ),legend=dict(
#            orientation="h"))
#py.iplot(dict(data=data, layout=layout), filename='basic-line')


