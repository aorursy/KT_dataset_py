import numpy as np

import pandas as pd
df = pd.DataFrame(np.random.randint(0, 20, 20).reshape(4,5), columns = 'c1 c2 c3 c4 c5'.split(), index = 'r1 r2 r3 r4'.split())

df
df[df%5==0] = np.nan

df
df.drop('r1')
df.drop('c3', axis=1)
# axis

# how

# thresh

# subset

# inplace
df.dropna()
df.dropna(axis=0)
df.dropna(axis=1)
df.dropna(how='any')
df.dropna(how='all')
df.dropna(subset=['c2'])
df.dropna(subset=['c3'])
df.dropna(subset=['c4'])
df.dropna(subset=['c3', 'c4'])
df.dropna(subset=['c2', 'c3'])
df.dropna(thresh=1, subset=['c3'])
# inplace
# value - scalar, dict, series, dataframe

# method

# axis

# inplace

# limit

# downcast
df
# filling with a scalar or constant value



df.fillna(value='new_val')
# filling with a dictionary



d = {'c1':1, 'c2':2, 'c3':3, 'c4':4, 'c5':5} # filling each column using coresponding column name

df.fillna(value=d)
# filling with a series



d = {'c1':1, 'c2':2, 'c3':3, 'c4':4, 'c5':5}

s = pd.Series(d)

print(s)

df.fillna(value=s)
# filling with a dataframe



d = {'c1':1, 'c2':2, 'c3':3, 'c4':4, 'c5':5}

df2 = pd.DataFrame(np.random.randint(0, 20, 20).reshape(4,5), 

                   columns = 'c1 c2 c3 c4 c5'.split(), 

                   index = 'r1 r2 r3 r4'.split())

print(df)

print(df2)

df.fillna(value=df2)
# pad / ffill: propagate last valid observation forward to next valid

# backfill / bfill: use next valid observation to fill gap.
df.fillna(method='ffill')
df.fillna(method='bfill')
df.fillna(method='pad')
df.fillna(method='backfill')