# import all libraries

import warnings # adds, removes or modifies python library behavior 
warnings.simplefilter(action='ignore', category=FutureWarning) # turn off future warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # turn off deprecation warnings
# Deprecation Warnings: cross_validation, weight_boosting, grid_search,learning_curve

import numpy as np # linear algebra
import pandas as pd # data processing
import logging # tracking events
import datetime # classes for dates
import time # time definitions
import os # operating system

import scipy.stats as stats # stats contains probability distributions
import pylab as pl # combines pyplot and numpy

import seaborn as sns # statistical data visualization
import matplotlib.pyplot as plt # 2D plotting library

# tools for data mining and data analysis
from sklearn import *

from xgboost import XGBClassifier # high performance gradient boosting
import lightgbm as lgb # fast, distributed, high performance gradient boosting

import plotly.offline as py # graphing library
py.init_notebook_mode(connected=True) # plot your graphs offline inside a Jupyter Notebook 
import plotly.graph_objs as go # web-service for hosting graphs
import plotly.tools as tls # web-service for hosting graphs
from kaggle.competitions import twosigmanews # imports kaggle module and create an environment
env = twosigmanews.make_env()
logging.info('Load data in 2 dataframes: mt_df (market_train_df) & nt_df (news_train_df)')
(mt_df, nt_df) = env.get_training_data()
days = env.get_prediction_days()
(mt_obs_df, nt_obs_df, predictions_template_df) = next(days)
print("market_train_df's shape:",mt_df.shape)
mt_df.dtypes
mt_df.isna().sum()
percent1 = (100 * mt_df.isnull().sum() / mt_df.shape[0]).sort_values(ascending=False)
percent1.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by market_obs_df", fontsize = 20)
mt_df.nunique()
mt_df.head(5)
mt_df.tail(5)
mt_df.select_dtypes(include=['float64']).describe()
mt_obs_df.head(5)
mt_obs_df.tail(5)
print('Oldest date:', mt_df['time'].min().strftime('%Y-%m-%d'))
print('Most recent date:', mt_df['time'].max().strftime('%Y-%m-%d'))
print('Total number of different dates:', mt_df['time'].nunique())
mt_df['time'].dt.time.describe()
mt_df["time"].groupby([mt_df["time"].dt.year, mt_df["time"].dt.month]).count().plot(kind="bar",figsize=(21,5))
print('Total number of unique assetCodes:', mt_df['assetCode'].nunique())
print(mt_df['assetCode'].values)
print('Total number of unique assetNames:', mt_df['assetName'].nunique())
print('Total number of unique assetCode & assetNames:', mt_df[['assetName','assetCode']].nunique())
print("There are {:,} records with assetName = `Unknown` in the training set".format(mt_df[mt_df['assetName'] == 'Unknown'].size))
assetNameGB = mt_df[mt_df['assetName'] == 'Unknown'].groupby('assetCode')
unknownAssets = assetNameGB.size().reset_index('assetCode')
print("There are {} unique assets without assetName in the training set".format(unknownAssets.shape[0]))
unknownAssets.columns = ['assetCode','unknowns']
unknownAssets.set_index('assetCode')
unknownAssets.loc[:15,['assetCode','unknowns']].sort_values(by='unknowns', ascending=False).head(10)

print(mt_df['assetName'].values)
mt_df['assetName'].iloc[0]
print('Min:', round(mt_df['volume'].min(),0))
print('Max:', round(mt_df['volume'].max(),0))
print('Mean:', round(mt_df['volume'].mean(),0))
print('Median:', round(mt_df['volume'].median(),0))
mt_df['volume'].plot(kind='hist', bins=[0,200000,400000,600000,800000,1000000]) 
mt_df['close'].describe().apply(lambda x: format(x, 'f'))
mt_df['open'].describe().apply(lambda x: format(x, 'f'))
mt_df['open']
mt_df['universe'].describe().apply(lambda x: format(x, 'f'))
# plotAsset plots assetCode1 from date1 to date2
def plotAsset(assetCode1,date1,date2):
    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                      & (mt_df['time'] > date1) 
                      & (mt_df['time'] < date2)]
    # Create a trace
    trace1 = go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values)

    layout = dict(title = "Closing prices of {}".format(assetCode1),
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),)
    
    data = [trace1]

    py.iplot(dict(data=data, layout=layout), filename='basic-line')
plotAsset('AAPL.O','2015-01-01','2017-01-01')
def Candlestick(assetCode1,date1,date2):

    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                  & (mt_df['time'] > date1) 
                  & (mt_df['time'] < date2)]
    
    asset_df['high'] = asset_df['open']
    asset_df['low'] = asset_df['close']

    for ind, row in asset_df.iterrows():
        if row['close'] > row['open']:
            asset_df.loc[ind, 'high'] = row['close']
            asset_df.loc[ind, 'low'] = row['open']

    trace1 = go.Candlestick(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        open = asset_df['open'].values,
        low = asset_df['low'].values,
        high = asset_df['high'].values,
        close = asset_df['close'].values
    )

    layout = dict(title = "Candlestick chart for {}".format(assetCode1),
                  xaxis = dict(
                      title = 'Month',
                      rangeslider = dict(visible = False)
                  ),
                  yaxis = dict(title = 'Price (USD)'))
    
    data = [trace1]

    py.iplot(dict(data=data, layout=layout), filename='basic-line')   
Candlestick('AAPL.O','2015-01-01','2017-01-01')
print("news_train_df's shape:",nt_df.shape) 
nt_df.dtypes
nt_df.isna().sum()
nt_df.nunique()
nt_df.head(5)
nt_df.tail(5)
nt_df.describe(include='all')
print('Oldest date:', nt_df['time'].min().strftime('%Y-%m-%d'))
print('Most recent date:', nt_df['time'].max().strftime('%Y-%m-%d'))
print("There are {} missing values in the `time` column".format(nt_df['time'].isna().sum()))
nt_df['time'].dt.date.describe()
print('Oldest date:', nt_df['sourceTimestamp'].min().strftime('%Y-%m-%d'))
print('Most recent date:', nt_df['sourceTimestamp'].max().strftime('%Y-%m-%d'))
print("There are {} missing values in the `sourceTimestamp` column".format(nt_df['sourceTimestamp'].isna().sum()))
nt_df['sourceTimestamp'].dt.date.describe()
print(nt_df.loc[nt_df['time'] == nt_df['sourceTimestamp']].shape[0])
print('Oldest date:', nt_df['firstCreated'].min().strftime('%Y-%m-%d'))
print('Most recent date:', nt_df['firstCreated'].max().strftime('%Y-%m-%d'))
print("There are {} missing values in the `firstCreated` column".format(nt_df['firstCreated'].isna().sum()))
nt_df['firstCreated'].dt.date.describe()
print("There are {} missing values in the `sourceId` column".format(nt_df.sourceId.isna().sum()))
print("There are {} unique values in the `sourceId` column".format(nt_df.sourceId.nunique()))
print("There are {} unique values in the `sourceId` column".format(nt_df.sourceId.count()))
print(nt_df.sourceId.describe())
nt_df[nt_df['sourceId']=='d7ad319ee02edea0']
nt_df[nt_df.duplicated(keep=False)].shape[0]
print("There are {} missing values in the `headline` column".format(nt_df['headline'].isna().sum()))
print("There are {} unique values in the `headline` column".format(nt_df.headline.nunique()))
print("There are {} unique values in the `headline` column".format(nt_df.headline.count()))
for i in range(0,20):
    print(nt_df['headline'].iloc[i])
print(nt_df['urgency'].describe())
print("Unique values in the `urgency` column: {}".format(nt_df['urgency'].unique()))
print(nt_df['urgency'].head(5))
print(nt_df.groupby(['urgency']).count())
sns.distplot(nt_df['urgency'])
print(nt_df['takeSequence'].head(5))
sns.distplot(nt_df['takeSequence'])
print(nt_df['provider'].head(5))
print(nt_df['subjects'].head(5))
for i in list(range(5)):
    print(nt_df['subjects'].iloc[i])
for i in list(range(5)):
    print(nt_df['audiences'].iloc[i])
print(nt_df['bodySize'].head(5))
print(nt_df['bodySize'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.companyCount)
print(nt_df['companyCount'].head(5))
sns.distplot(nt_df.companyCount)
print("There are {} missing values in the `headlineTag` column".format(nt_df['headlineTag'].isna().sum()))
print("There are {} unique values in the `headlineTag` column".format(nt_df.headlineTag.nunique()))
print("There are {} unique values in the `headlineTag` column".format(nt_df.headlineTag.count()))
print(nt_df['headlineTag'].unique())
nt_df['marketCommentary'].unique()
print(nt_df['sentenceCount'].head(5))
print(nt_df['sentenceCount'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentenceCount)
print(nt_df['wordCount'].head(5))
print(nt_df['wordCount'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.wordCount)
print(nt_df['assetCodes'].head(5))
print(nt_df['assetName'].head(5))
print(nt_df['firstMentionSentence'].head(5))
print(nt_df['firstMentionSentence'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.firstMentionSentence)
print(nt_df['relevance'].head(5))
print(nt_df['relevance'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.relevance)
print(nt_df['sentimentClass'].unique())
print(nt_df['sentimentClass'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentClass)
print(nt_df['sentimentNegative'].unique())
print(nt_df['sentimentNegative'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentNegative)
print(nt_df['sentimentNeutral'].unique())
print(nt_df['sentimentNeutral'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentNeutral)
print(nt_df['sentimentPositive'].head(5))
print(nt_df['sentimentPositive'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentNeutral)
print(nt_df['sentimentWordCount'].head(5))
print(nt_df['sentimentWordCount'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentWordCount)
print(nt_df['noveltyCount12H'].head(5))
print(nt_df['noveltyCount12H'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.noveltyCount12H)
# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
# days = env.get_prediction_days()
# (market_obs_df, news_obs_df, predictions_template_df) = next(days)
# predictions_template_df.head()
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')
env.write_submission_file()
market_train = mt_df
news_train = nt_df
market_train.time = market_train.time.dt.date
news_train.time = news_train.time.dt.hour
news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
news_train.firstCreated = news_train.firstCreated.dt.date
news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))
news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
kcol = ['firstCreated', 'assetCodes']
news_train = news_train.groupby(kcol, as_index=False).mean()

market_train = pd.merge(market_train, news_train, 
                        how='left', 
                        left_on=['time', 'assetCode'], 
                        right_on=['firstCreated', 'assetCodes'])

lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}

market_train['assetCodeT'] = market_train['assetCode'].map(lbl)

fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 
                                             'assetName', 'audiences', 'firstCreated', 
                                             'headline', 'headlineTag', 'marketCommentary', 
                                             'provider', 'returnsOpenNextMktres10', 'sourceId',
                                             'subjects', 'time', 'time_x', 'universe']]
x1, x2, y1, y2 = model_selection.train_test_split(market_train[fcol], 
                                                  market_train['returnsOpenNextMktres10'], 
                                                  test_size=0.25, 
                                                  random_state=99)

def lgb_rmse(preds, y): # update to Competition Metric
    y = np.array(list(y.get_label()))
    score = np.sqrt(metrics.mean_squared_error(y, preds))
    return 'RMSE', score, False

params = {'learning_rate': 0.2, 
          'max_depth': 6, 
          'boosting': 'gbdt',
          'objective': 'regression',
          'seed': 2018}

lgb_model = lgb.train(params, 
                      lgb.Dataset(x1, label=y1), 
                      500, 
                      lgb.Dataset(x2, label=y2), 
                      verbose_eval=10,
                      early_stopping_rounds=20)
df = pd.DataFrame({'imp': lgb_model.feature_importance(importance_type='gain'), 'col':fcol})
df = df.sort_values(['imp','col'], ascending=[True, False])
_ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))
#plt.savefig('lgb_gain.png')

df = pd.DataFrame({'imp': lgb_model.feature_importance(importance_type='split'), 'col':fcol})
df = df.sort_values(['imp','col'], ascending=[True, False])
_ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))
# plt.savefig('lgb_split.png')
env.write_submission_file()
