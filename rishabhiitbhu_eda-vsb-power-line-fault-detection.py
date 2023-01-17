import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os
from matplotlib import pyplot as plt
%matplotlib inline
train = pq.read_pandas('../input/train.parquet').to_pandas()
train.info()
plt.figure(figsize=(24, 8))
plt.plot(train.iloc[:, :5]);
meta = pd.read_csv('../input/metadata_train.csv')
meta.info()
meta.describe()
meta.corr()
meta.head(10)
# get positive and negative `id_measurement`s
positive_mid = np.unique(meta.loc[meta.target == 1, 'id_measurement'].values)
negative_mid = np.unique(meta.loc[meta.target == 0, 'id_measurement'].values)
# get one positive and one negative signal_id
pid = meta.loc[meta.id_measurement == positive_mid[0], 'signal_id']
nid = meta.loc[meta.id_measurement == negative_mid[0], 'signal_id']
positive_sample = train.iloc[:, pid]
negative_sample = train.iloc[:, nid]
plt.figure(figsize=(24, 8))
plt.plot(positive_sample);
plt.figure(figsize=(24, 8))
plt.plot(negative_sample);
meta['target'].value_counts().plot(kind='bar');
meta['target'].value_counts()
meta['phase'].value_counts()

