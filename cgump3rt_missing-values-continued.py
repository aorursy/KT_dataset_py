import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import re

import seaborn as sns

%matplotlib inline
with pd.HDFStore('../input/train.h5') as train:

    df = train.get('train')
# get total number of NaN values per feature column for every timestamp

nans = df.groupby('timestamp').apply(lambda x: x.isnull().sum())

# get change with respect 

nans = nans.diff().drop(['id','timestamp','y'],axis=1)

nans.plot(figsize=(13,10))

plt.legend(ncol=5,loc='best');
# get the number of assets per timestamp

nassets = df.groupby('timestamp').apply(len)

# get the change with respect to the previous timestamp

delta_assets = nassets.diff()



# plot again

nans.plot(figsize=(13,10))

delta_assets.plot(style=['.b'],ax=plt.gca(),label='number of assets')

plt.legend(ncol=5,loc='best');
# get number of columns for each group

n_fundamental = len([g for g in nans.columns.tolist() if re.search("fundamental_*",g)])

n_derived = len([g for g in nans.columns.tolist() if re.search("derived_*",g)])

n_technical = len([g for g in nans.columns.tolist() if re.search("technical_*",g)])
nans['fundamental'] = nans.filter(regex='fundamental_*').sum(axis=1)/nassets/n_fundamental

nans['derived'] = nans.filter(regex='derived_*').sum(axis=1)/nassets/n_derived

nans['technical'] = nans.filter(regex='technical_*').sum(axis=1)/nassets/n_technical

nans[['fundamental','derived','technical']].plot(figsize=(12,6));
max_lag = 100

corrs = np.zeros((max_lag,nans.shape[1]))

for l in range(1,max_lag):

    c = delta_assets.shift(l)

    corrs[l] = nans.corrwith(c)



plt.figure(figsize=(15,9))

sns.heatmap(corrs,xticklabels=nans.columns.tolist(),linewidths=0.05,linecolor='gray')

plt.ylabel('lag');
# group by asset ID and get number of missing values for each feature column

ids = sorted(df.id.unique())

columns = df.columns.drop(['id','timestamp','y']).insert(0,'length')

nan_df = pd.DataFrame(data=None,index=ids,columns=columns,dtype=float)

# iterate over all asset ID

for name,group in df.groupby('id'):

    # for every feature column

    for c in columns:

        if c == 'length':

            nan_df.loc[name,c] = int(len(group))

        else:

            # total number of rows with missing data

            nan_df.loc[name,c] = float(group[c].isnull().sum())
nan_df.head()
# truncate all numbers at 100

capped = nan_df.copy()

capped[capped > 100] = 100

capped = capped.div(capped['length'],axis='index').drop(['length'],axis=1)

capped.head()
from sklearn.cluster import dbscan

_,labels = dbscan(capped.values)

plt.hist(labels,bins=25,range=(-0.5,24.5));