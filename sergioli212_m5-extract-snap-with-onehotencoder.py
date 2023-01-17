import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
# Conditional statements(Slow)
# def get_SNAP_for_store(row):
#     print(row)
#     if row['state_id'] == 'CA':
#         return row['snap_CA']
#     if row['state_id'] == 'TX':
#         return row['snap_TX']
#     if row['state_id'] == 'WI':
#         return row['snap_WI']
# grid_df['snap'] = grid_df.apply(lambda row: get_SNAP_for_store(row), axis=1)
grid_df = pd.read_pickle('../input/m5-sample50/grid_df50.pkl')
grid_df = grid_df[['snap_CA', 'snap_TX', 'snap_WI', 'state_id']]
grid_df.head()
enc = preprocessing.OneHotEncoder()
state = enc.fit_transform(grid_df[['state_id']]).toarray()
state[:5]
grid_df['snap'] = np.multiply(grid_df[['snap_CA', 'snap_TX', 'snap_WI']].values,state).sum(axis=1)
grid_df