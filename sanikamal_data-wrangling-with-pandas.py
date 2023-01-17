import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
my_series = pd.Series([1,3,5,np.nan,6,8,10,23,np.nan,89])
my_series
dates = pd.date_range('20180917',periods=15)
dates
df = pd.DataFrame(np.random.randn(15,5), index=dates, columns={'donkey','tiger','lion','monkey','cat'})
df
df2 = pd.DataFrame({ 'A' : 1.,
'B' : pd.Timestamp('20180917'),
'C' : pd.Series(1,index=list(range(7)),dtype='float32'),
'D' : np.array([3] * 7,dtype='int32'),
'E' : pd.Categorical(["test","train","test","train","train","test","train"]),
'F' : 'foo' })
df2
df2.dtypes
print(df.index)
print(df.columns)
print(df.values)
df.describe()
df.T
df.sort_values(by='cat')
df.sort_index(axis=0, ascending=False)
df.sort_index(axis=1)
df.head()
df.tail()
df.tail(3)
df['tiger']
df[1:6]
df['20180920':'20180929']
df.loc[:,['cat','tiger']]
df.loc['20180921':'20180929',['cat','tiger']]
df[df['cat'] < 0]
neg_only = df[df < 0]
neg_only
neg_only.dropna()
neg_only.fillna(value=0)
df2 = df.copy()
df2['color']=['blue', 'green','red','blue','green','red','blue','blue', 'green','red','blue','green','red','blue','red']
df2
df2[df2['color'].isin(['green','red'])]
df.mean()
df.mean(1)
df.median()
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())
frame_one = pd.DataFrame(np.random.randn(5, 4))
frame_one
frame_two = pd.DataFrame(np.random.randn(5, 4))
frame_two
pd.concat([frame_one, frame_two])
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
left
right = pd.DataFrame({'key': ['foo', 'foo', 'bar'], 'rval': [3, 4, 5]})
right
pd.merge(left, right, on='key')
foo_bar = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
'C' : np.random.randn(8),
'D' : np.random.randn(8)})
foo_bar
foo_bar.groupby('A').sum()
grouped = foo_bar.groupby(['A','B']).sum()
grouped
stacked = grouped.stack()
stacked
stacked.unstack()
rng = pd.date_range('19/10/2017', periods=200, freq='S')
rng
time_series = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
time_series.head()
time_series.resample('1Min').sum()
ts_utc = time_series.tz_localize('UTC')
ts_utc.head()
ts_utc.tz_convert('America/New_York').head()
ts_utc.to_csv('ts_utc.csv')
new_frame = pd.read_csv('ts_utc.csv')
new_frame.head()
new_frame.to_excel('utc.xlsx', sheet_name='Sheet1')
pd.read_excel('utc.xlsx', 'Sheet1', index_col=None, na_values=['NA']).head()