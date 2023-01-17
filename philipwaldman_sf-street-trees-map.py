import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/sf-street-trees/san_francisco_street_trees.csv')

df.head(3)
df['longitude'].describe()
df['latitude'].describe()
mask = (df['longitude'] > -125.0) & ((df['latitude'] > 37.6) & (df['latitude'] < 40.0))

df_sf = df[mask]

cols = ['red' if i=='Private' else 'blue' for i in df_sf['care_taker']]

df_sf.plot(x='longitude', y='latitude', kind='scatter', c=cols, s=0.05, figsize=(15, 10));