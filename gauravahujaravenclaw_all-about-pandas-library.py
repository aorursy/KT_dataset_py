import pandas as pd

import numpy as np
s = pd.Series([1 , 3, 5, np.nan, 6, 8])

s
dates = pd.date_range('20130101' , periods=6)

dates
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns= list('ABCD'))

df
df2 =pd.DataFrame({'A':1.,

              'B': pd.Timestamp('20130102'),

              'C': pd.Series(1, index=list(range(4)), dtype='float32'),

              'D': np.array([3] * 4, dtype='int32'),

              'E': pd.Categorical(["test", "train", "test", "train"]),

              'F': 'foo'})



df2
df2.dtypes
df.head()
df.tail(3)
df.index
df.columns
df.to_numpy()
df2.to_numpy()
df.describe()
df.T
df.sort_index(axis = 1, ascending=False)
df.sort_values(by='C')
df['A']
df[0:3]
df['20130102':'20130104']
df.loc[dates[0]]
df.loc[:,['A','B']]
df.loc['20130102':'20130104',['A','B']]
df.loc[dates[0], 'A']
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102' , periods=6))

s1
df['F'] = s1

df
df.loc[:, 'D'] = np.array([5]*len(df))

df
df2 = df.copy()



df2[df2 > 0] = -df2



df2
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1

df1
df1.dropna(how='any')
df1.fillna(value=5)
pd.isna(df1)
df.mean()
df.mean(1)
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())
df = pd.DataFrame(np.random.randn(10, 4))

df
pieces = [df[:3], df[3:7], df[7:]]



pd.concat(pieces)
df = pd.DataFrame({'A': ['foo' , 'bar','foo' , 'bar','foo' , 'bar','foo' , 'foo'],

                   'B' :['one', 'one', 'two', 'three','one', 'two', 'three','one'],

                   'C' :np.random.randn(8),

                   'D' :np.random.randn(8)})



df
df.groupby('A').sum()
df.groupby(['A','B']).sum()
df = pd.DataFrame({'A': ['foo' , 'bar','foo' , 'bar','foo' , 'bar','foo' , 'foo'],

                   'B': ['A','B','C','D'] * 2,

                   'C' :['one', 'one', 'two', 'three','one', 'two', 'three','one'],

                   'D' :np.random.randn(8),

                   'E' :np.random.randn(8)})

df
pd.pivot_table(df, values='D', index=['A','B'], columns=['C'])