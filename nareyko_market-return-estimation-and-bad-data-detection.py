import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kaggle.competitions import twosigmanews

# Read data but just once
if 'env' not in globals():
    env = twosigmanews.make_env()
    (market_train_df, news_train_df) = env.get_training_data()
df = market_train_df[[
    'time',
    'assetCode',
    'open',
    'returnsOpenPrevRaw1', 
    'returnsOpenPrevMktres1',
    'returnsOpenNextMktres10',
]].copy()
df = df[~df.returnsOpenPrevMktres1.isnull() & ~df.returnsOpenPrevRaw1.isnull()]
df['returnsOpenPrevMktres1_abs'] = df['returnsOpenPrevMktres1'].abs()
df['returnsOpenPrevMktres1_abs_min'] = df.groupby('time').returnsOpenPrevMktres1_abs.transform('min')
df_min = df[df.returnsOpenPrevMktres1_abs == df.returnsOpenPrevMktres1_abs_min]
df_min.head()
(df_min.groupby('time').size()==1).all()
df_min.groupby('time').returnsOpenPrevRaw1.mean().plot(figsize=(15,6))
for y in df_min.time.dt.year.unique():
    df_min[df_min.time.dt.year==y].groupby('time').returnsOpenPrevRaw1.mean().plot(title=str(y), figsize=(15,6))
    plt.show()
df_min[(df_min.time>='2016-07-01') & (df_min.time<'2016-08-01')]
df_min[df_min.returnsOpenPrevRaw1==0].groupby('assetCode').size().sort_values(ascending=False).head(10)
market_train_df[market_train_df.assetCode.isin(['PGN.N', 'EBRYY.OB'])].groupby('assetCode').size()
market_train_df[market_train_df.assetCode.isin(['PGN.N', 'EBRYY.OB'])].groupby('assetCode').open.nunique()
market_train_df[market_train_df.assetCode == 'PGN.N']
market_train_df[market_train_df.assetCode == 'EBRYY.OB']
market_train_df[market_train_df.assetName=='Unknown'].assetCode.unique()
df_filtered = df[~df.assetCode.isin(market_train_df[market_train_df.assetName=='Unknown'].assetCode.unique())]
df_filtered = df_filtered[df_filtered.assetCode != 'PGN.N']
df_filtered['returnsOpenPrevMktres1_abs'] = df_filtered['returnsOpenPrevMktres1'].abs()
df_filtered['returnsOpenPrevMktres1_abs_min'] = df_filtered.groupby('time').returnsOpenPrevMktres1_abs.transform('min')
df_min_filtered = df_filtered[df_filtered.returnsOpenPrevMktres1_abs == df_filtered.returnsOpenPrevMktres1_abs_min]
df_min_filtered.groupby('time').returnsOpenPrevRaw1.mean().plot(figsize=(15,6))
for y in df_min_filtered.time.dt.year.unique():
    df_min_filtered[df_min_filtered.time.dt.year==y].groupby('time').returnsOpenPrevRaw1.mean().plot(title=str(y), figsize=(15,6))
    plt.show()
