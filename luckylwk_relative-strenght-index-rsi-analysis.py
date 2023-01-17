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
df[df.id==ids[4]]['y-stock'].plot(figsize=(9,2))
if 'y-delta' not in df.columns:

    df['y-delta'] = np.nan





for assetId in ids:

    ix = df.id == assetId

    df.loc[ix, 'y-delta'] = df.loc[ix, 'y-stock']-df.loc[ix, 'y-stock'].shift(1)
df[df.id==ids[4]]['y-delta'].plot(figsize=(9,1))
df[df.id==ids[4]]['y'].plot(figsize=(9,1))
if 'RSI-U' not in df.columns:

    df['RSI-U'] = np.nan

    df['RSI-D'] = np.nan 

    df['U-EWA-14'] = np.nan

    df['D-EWA-14'] = np.nan

    df['RS'] = np.nan

    df['RSI'] = np.nan





def calculate_ema(series_y, size=30):

    return series_y.ewm(span=size, min_periods=size).mean()





for assetId in ids:

    ix = df.id == assetId

    upIx = df.loc[ix, 'y-delta'] > 0.0

    downIx = df.loc[ix, 'y-delta'] < 0.0

    upIx = upIx.index[upIx.values]

    downIx = downIx.index[downIx.values]

    df.ix[upIx, 'RSI-U'] = df.ix[upIx, 'y-delta']

    df.ix[downIx, 'RSI-U'] = 0.0

    df.ix[upIx, 'RSI-D'] = 0.0

    df.ix[downIx, 'RSI-D'] = np.abs(df.ix[downIx, 'y-delta'])

    df.loc[ix, 'U-EWA-14'] = calculate_ema(df.loc[ix, 'RSI-U'], size=14)

    df.loc[ix, 'D-EWA-14'] = calculate_ema(df.loc[ix, 'RSI-D'], size=14)

    df.loc[ix, 'RS'] = df.loc[ix, 'U-EWA-14'] / df.loc[ix, 'D-EWA-14']

    df.loc[ix, 'RSI'] = 100 - (100 / (1 + df.loc[ix, 'RS']))
plt.figure(figsize=(9,4))

plt.subplot(2,1,1)

df[df.id==ids[4]]['y-stock'].plot()

plt.subplot(2,1,2)

df[df.id==ids[4]]['RSI'].plot()
correlations = pd.DataFrame()



feat_cols = df.columns[2:-7]



for col in df.columns[-7:]:

    corrs = []

    for f_col in feat_cols:

        corrs.append( df.loc[df[col].notnull(), col].corr(df.loc[df[col].notnull(), f_col]) )

    correlations[col] = corrs

    

# Set index to columns.

correlations.set_index(feat_cols, inplace=True)
import seaborn as sns



plt.figure(figsize=(8,15))

sns.heatmap(correlations, vmin=-1.0, vmax=1.0)