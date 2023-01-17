from kaggle.competitions import twosigmanews
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain

%matplotlib inline

# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train_df = market_train_df.tail(100_000)
news_train_df = news_train_df.tail(300_000)
#market_train_df.shape, news_train_df.shape
news_train_df['time'] = (news_train_df['time'] - np.timedelta64(22,'h')).dt.ceil('1D')
#news_train_df['time']
market_train_df['time'] = market_train_df['time'].dt.floor('1D')
news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'") 
assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
assetCodes_index = news_train_df.index.repeat( news_train_df['assetCodes'].apply(len) )
assert len(assetCodes_index) == len(assetCodes_expanded)
df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

df_assetCodes.head(5)
news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean', 'std'],
    'wordCount': ['min', 'max', 'mean', 'std'],
    'sentenceCount': ['min', 'max', 'mean', 'std'],
    'companyCount': ['min', 'max', 'mean', 'std'],
    'marketCommentary': ['min', 'max', 'mean', 'std'],
    'relevance': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
    'sentimentWordCount': ['min', 'max', 'mean', 'std'],
    'noveltyCount12H': ['min', 'max', 'mean', 'std'],
    'noveltyCount24H': ['min', 'max', 'mean', 'std'],
    'noveltyCount3D': ['min', 'max', 'mean', 'std'],
    'noveltyCount5D': ['min', 'max', 'mean', 'std'],
    'noveltyCount7D': ['min', 'max', 'mean', 'std'],
    'volumeCounts12H': ['min', 'max', 'mean', 'std'],
    'volumeCounts24H': ['min', 'max', 'mean', 'std'],
    'volumeCounts3D': ['min', 'max', 'mean', 'std'],
    'volumeCounts5D': ['min', 'max', 'mean', 'std'],
    'volumeCounts7D': ['min', 'max', 'mean', 'std']
}
news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))
news_train_df_expanded.head()
del news_train_df, df_assetCodes
news_train_df_aggregated = news_train_df_expanded.groupby(['time', 'assetCode']).agg(news_cols_agg)
del news_train_df_expanded

news_train_df_aggregated.head(10)
news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)
news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]

news_train_df_aggregated.head(10)
market_train_df.head()
market_train_df = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])

market_train_df.head()
del news_train_df_aggregated
x = market_train_df
#del market_train_df

x.head()
def label_encode(series, min_count):
    vc = series.value_counts()
    le = {c:i for i, c in enumerate(vc.index[vc >= min_count])}
    return le

le_assetCode = label_encode(x['assetCode'], min_count=10)
le_assetName = label_encode(x['assetName'], min_count=5)
x['assetCode'] = x['assetCode'].map(le_assetCode).fillna(-1).astype(int)
x['assetName'] = x['assetName'].map(le_assetName).fillna(-1).astype(int)
x.head()
x.drop(columns=['returnsOpenNextMktres10'], inplace=True)
x.drop(columns=['universe'], inplace=True)
x['dayofweek'], x['month'] = x.time.dt.dayofweek, x.time.dt.month
x.drop(columns='time', inplace=True)

x.head()
for bogus_col in ['marketCommentary_min', 'marketCommentary_max']:
    x[bogus_col] = x[bogus_col].astype(float)
le = (le_assetCode, le_assetName)
y = market_train_df['returnsOpenNextMktres10'].clip(-1, 1)
