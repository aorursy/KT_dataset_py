import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.random import randn
np.random.seed(101)
df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])
df
df['W']
type(df['W'])
type(df)
df[['W', 'Z']]
df['new'] = df['W'] + df['Y']
df
df.drop('new', axis = 1, inplace=True)
df
df.drop('E', axis = 0)
df.shape
df
df.loc['C']
df.iloc[2]
df.loc['B','Y']
df.loc[['A', 'B'],['W', 'Y']]
df
booldf = df > 0 
df[booldf]
df['W'] > 0
df[df['W']>0]
df[df['W'] < 0]
df[df['W'] > 0]['X']
df[df['W']>0][['Y','X']]
boolser = df['W']>0

result = df[boolser]

result[['Y', 'X']]
df[(df['W']>0) & (df['Y'] > 1)]
df.reset_index()
newind = 'CA NY WY OR CO'.split()
newind
df['States'] = newind
df
df.set_index('States', inplace = True)
df
outside = ['G1' , 'G1', 'G1', 'G2', 'G2', 'G2']

inside = [1, 2, 3, 1, 2, 3]

hier_index = list(zip(outside, inside))

hier_index = pd.MultiIndex.from_tuples(hier_index)
df = pd.DataFrame(randn(6, 2), hier_index,['A', 'B'])
df.index.names = ['Groups', 'Num']
df
df.xs(1, level='Num')