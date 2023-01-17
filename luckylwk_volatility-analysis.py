import pandas as pd

import numpy as np



import matplotlib.pyplot as plt
with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")
print(df.shape) # Size of the dataset

print(df.timestamp.unique().shape[0]) # How many distinct days?

print(df.id.unique().shape[0]) # How many distinct assets?
plt.figure(figsize=(9,2))

df.groupby('id').size().hist(bins=200) # How many days per asset.
# Lets take all assets with at least 100 days.

min_days = 100



ids_tmp = df.groupby('id').size() > min_days

print(ids_tmp.shape)

ids = ids_tmp.index.values[np.where(ids_tmp.values==True)]

print(ids.shape)



ids_tmp = None
# For these assets we will calculate the 30 day rolling volatility.

df['Vol-20d'] = np.nan

df['Vol-30d'] = np.nan

df['Vol-60d'] = np.nan

df['Vol-90d'] = np.nan





# Function to calculate the annualised volatility

def calculate_volatility(series_y, size=30):

    return np.sqrt(260) * series_y.rolling(window=size, min_periods=size, center=False).std()





for assetId in ids:

    ix = df.id == assetId

    df.loc[ix, 'Vol-20d'] = calculate_volatility(df.loc[ix, 'y'], size=20)

    df.loc[ix, 'Vol-30d'] = calculate_volatility(df.loc[ix, 'y'], size=30)

    df.loc[ix, 'Vol-60d'] = calculate_volatility(df.loc[ix, 'y'], size=60)

    df.loc[ix, 'Vol-90d'] = calculate_volatility(df.loc[ix, 'y'], size=90)



print('Done!')
# Lets plot the volatility versus the returns...

plt.figure(figsize=(9,2))

df[df['Vol-20d'].notnull()].plot(x='Vol-20d', y='y', kind='scatter', alpha=0.0025)
plt.figure(figsize=(9,2))

df[df['Vol-30d'].notnull()].plot(x='Vol-30d', y='y', kind='scatter', alpha=0.0025)
plt.figure(figsize=(9,2))

df[df['Vol-60d'].notnull()].plot(x='Vol-60d', y='y', kind='scatter', alpha=0.0025)
plt.figure(figsize=(9,2))

df[df['Vol-90d'].notnull()].plot(x='Vol-90d', y='y', kind='scatter', alpha=0.0025)
df.loc[:,df.columns[-4:]].describe()
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