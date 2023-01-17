import pandas as pd

import numpy as np



import matplotlib.pyplot as plt
with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")
# Lets take all assets with at least 100 days.

min_days = 100



ids_tmp = df.groupby('id').size() > min_days

print(ids_tmp.shape)

ids = ids_tmp.index.values[np.where(ids_tmp.values==True)]

print(ids.shape)



ids_tmp = None
if 'y-stock' not in df.columns:

    df['y-stock'] = np.nan





for assetId in ids:

    ix = df.id == assetId

    df.loc[ix, 'y-stock'] = 100 * np.cumprod(1.0 + df[ix].y)
if 'EMA-9d' not in df.columns:

    df['EMA-12d'] = np.nan

    df['EMA-26d'] = np.nan





def calculate_ema(series_y, size=30):

    return series_y.ewm(span=size, min_periods=size).mean()





for assetId in ids:

    ix = df.id == assetId

    df.loc[ix, 'EMA-12d'] = calculate_ema(df.loc[ix, 'y-stock'], size=12)

    df.loc[ix, 'EMA-26d'] = calculate_ema(df.loc[ix, 'y-stock'], size=26)
df.loc[df.id==ids[0], ['EMA-9d', 'EMA-12d', 'EMA-26d', 'y-stock']].plot(figsize=(9,2))
# Moving Average Convergence Divergence (MACD)

# https://en.wikipedia.org/wiki/MACD



if 'MACD-diff' not in df.columns:

    df['MACD-diff'] = np.nan

    df['MACD(9,12,26)'] = np.nan

    



for assetId in ids:

    ix = df.id == assetId

    df.loc[ix, 'MACD-diff'] = df.loc[ix, 'EMA-12d'] - df.loc[ix, 'EMA-26d']

    df.loc[ix, 'MACD(9,12,26)'] = calculate_ema(df.loc[ix, 'MACD-diff'], size=9)
plt.figure(figsize=(9,5))

ax = plt.subplot(2,1,1)

df.loc[df.id==ids[0], ['y-stock']].plot(ax=ax)

ax = plt.subplot(2,1,2)

df.loc[df.id==ids[0], ['MACD-diff', 'MACD(9,12,26)']].plot(ax=ax)
correlations = pd.DataFrame()



feat_cols = df.columns[2:-5]



for col in df.columns[-4:]:

    corrs = []

    for f_col in feat_cols:

        corrs.append( df.loc[df[col].notnull(), col].corr(df.loc[df[col].notnull(), f_col]) )

    correlations[col] = corrs

    

# Set index to columns.

correlations.set_index(feat_cols, inplace=True)
import seaborn as sns



plt.figure(figsize=(8,15))

sns.heatmap(correlations, vmin=-1.0, vmax=1.0)