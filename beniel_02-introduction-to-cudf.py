!nvidia-smi
!nvcc --version
import sys

!rsync -ah --progress ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!rsync -ah --progress /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf; print('cuDF Version:', cudf.__version__)
column = cudf.Series([10, 11, 12, 13])

column
print(column)
print(column.index)
new_column = column.set_index([5, 6, 7, 8]) 

print(new_column)
df = cudf.DataFrame()

print(df)
import numpy as np; print('NumPy Version:', np.__version__)





# here we create two columns named "key" and "value"

df['key'] = [0, 1, 2, 3, 4]

df['value'] = np.arange(10, 15)

print(df)
from datetime import datetime, timedelta





ids = np.arange(5)

t0 = datetime.strptime('2018-10-07 12:00:00', '%Y-%m-%d %H:%M:%S')

timestamps = [(t0+ timedelta(seconds=x)) for x in range(5)]

timestamps_np = np.array(timestamps, dtype='datetime64')
df = cudf.DataFrame()

df['ids'] = ids

df['timestamp'] = timestamps_np

print(df)
df = cudf.DataFrame({'id': ids, 'timestamp': timestamps_np})

print(df)
import pandas as pd; print('Pandas Version:', pd.__version__)





pandas_df = pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

                          'b': [0.0, 0.1, 0.2, None, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})

print(pandas_df)
df = cudf.from_pandas(pandas_df)

# df = cudf.DataFrame.from_pandas(pandas_df)  # alternative

print(df)
column1 = cudf.Series([1, 2, 3, 4])

column2 = cudf.Series([5, 6, 7, 8])

column3 = cudf.Series([9, 10, 11, 12])

df = cudf.DataFrame({'a': column1, 'b': column2, 'c': column3})

print(df)
df = cudf.DataFrame({'a': np.arange(0, 100), 'b': np.arange(100, 0, -1)})
df
print(df)
print(df.head())
print(df.columns)
df.columns = ['c', 'd']

print(df.columns)
print(df.dtypes)
df['c'] = df['c'].astype(np.float32)

df['d'] = df['d'].astype(np.int32)

print(df.dtypes)
print(type(df['c']))

print(df['c'])
df.index
print(df[df.index == 2])
pandas_df = df.to_pandas()

print(type(pandas_df))
numpy_array = df.to_pandas().values

print(type(numpy_array))
df.to_pandas().to_csv('./dataset.csv', index=False)
df = cudf.read_csv('./dataset.csv')

print(df)
df = cudf.DataFrame({'a': np.arange(0, 100).astype(np.float32), 

                     'b': np.arange(100, 0, -1).astype(np.float32)})
print(df[0:5])
print(df['a'])

# print(df.a)  # alternative
print(df[['a', 'b']])
print(df.loc[0:5, ['a']])

# print(df.loc[0:5, ['a', 'b']])  # to select multiple columns, pass in multiple column names
df = cudf.DataFrame({'a': np.arange(0, 100).astype(np.float32), 

                     'b': np.arange(100, 0, -1).astype(np.float32), 

                     'c': np.arange(100, 200).astype(np.float32)})
df['d'] = np.arange(200, 300).astype(np.float32)



print(df)
data = np.arange(300, 400).astype(np.float32)

df.add_column('e', data)



print(df)
df = cudf.DataFrame({'a': np.arange(0, 100).astype(np.float32), 

                     'b': np.arange(100, 0, -1).astype(np.float32), 

                     'c': np.arange(100, 200).astype(np.float32)})
df.drop_column('a')

print(df)
df = cudf.DataFrame({'a': np.arange(0, 100).astype(np.float32), 

                     'b': np.arange(100, 0, -1).astype(np.float32), 

                     'c': np.arange(100, 200).astype(np.float32)})
new_df = df.drop('a')



print('Original DataFrame:')

print(df)

print(79 * '-')

print('New DataFrame:')

print(new_df)
new_df = df.drop(['a', 'b'])



print('Original DataFrame:')

print(df)

print(79 * '-')

print('New DataFrame:')

print(new_df)
df = cudf.DataFrame({'a': [0, None, 2, 3, 4, 5, 6, 7, 8, None, 10],

                     'b': [0.0, 0.1, 0.2, None, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 

                     'c': [0.0, 0.1, None, None, 0.4, 0.5, None, 0.7, 0.8, 0.9, 1.0]})

print(df)
df['c'] = df['c'].fillna(999)

print(df)
new_df = df.fillna(-1)

print(new_df)
df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                     'b': np.random.randint(2, size=100).astype(np.int32), 

                     'c': np.arange(0, 100).astype(np.int32), 

                     'd': np.arange(100, 0, -1).astype(np.int32)})
mask = df['a'] == 3

mask
df[mask]
df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                     'b': np.random.randint(2, size=100).astype(np.int32), 

                     'c': np.arange(0, 100).astype(np.int32), 

                     'd': np.arange(100, 0, -1).astype(np.int32)})

print(df.head())
print(df.sort_values('d').head())
print(df.sort_values('c', ascending=False).head())
print(df.sort_values(['a', 'b']).head())
print('Sort with all columns specified descending:')

print(df.sort_values(['a', 'b'], ascending=False).head())

print(79 * '-')

print('Sort with both a descending and b ascending:')

print(df.sort_values(['a', 'b'], ascending=[False, True]).head())
df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                     'b': np.random.randint(2, size=100).astype(np.int32), 

                     'c': np.arange(0, 100).astype(np.int32), 

                     'd': np.arange(100, 0, -1).astype(np.int32)})
df['a'].sum()
print(df.sum())
df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                     'b': np.random.randint(2, size=100).astype(np.int32), 

                     'c': np.arange(0, 100).astype(np.int32), 

                     'd': np.arange(100, 0, -1).astype(np.int32)})
def add_ten_to_x(x):

    return x + 10



print(df['c'].applymap(add_ten_to_x))
df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                     'b': np.random.randint(2, size=100).astype(np.int32), 

                     'c': np.arange(0, 100).astype(np.int32), 

                     'd': np.arange(100, 0, -1).astype(np.int32)})
result = df['a'].value_counts()

print(result)
df1 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                      'b': np.random.randint(2, size=100).astype(np.int32), 

                      'c': np.arange(0, 100).astype(np.int32), 

                      'd': np.arange(100, 0, -1).astype(np.int32)})

df2 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                      'b': np.random.randint(2, size=100).astype(np.int32), 

                      'c': np.arange(0, 100).astype(np.int32), 

                      'd': np.arange(100, 0, -1).astype(np.int32)})
df = cudf.concat([df1, df2], axis=0)

df
df1 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                      'b': np.random.randint(2, size=100).astype(np.int32), 

                      'c': np.arange(0, 100).astype(np.int32), 

                      'd': np.arange(100, 0, -1).astype(np.int32)})

df2 = cudf.DataFrame({'e': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                      'f': np.random.randint(2, size=100).astype(np.int32), 

                      'g': np.arange(0, 100).astype(np.int32), 

                      'h': np.arange(100, 0, -1).astype(np.int32)})
df = cudf.concat([df1, df2], axis=1)

df
df1 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                      'b': np.random.randint(2, size=100).astype(np.int32), 

                      'c': np.arange(0, 100).astype(np.int32), 

                      'd': np.arange(100, 0, -1).astype(np.int32)})

df2 = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                      'b': np.random.randint(2, size=100).astype(np.int32), 

                      'e': np.arange(0, 100).astype(np.int32), 

                      'f': np.arange(100, 0, -1).astype(np.int32)})
df = df1.merge(df2)

print(df.head())
df = df1.merge(df2, on=['a'])

print(df.head())
df = df1.merge(df2, on=['a', 'b'])

print(df.head())
df = cudf.merge(df1, df2)

print(df.head())
df = cudf.merge(df1, df2, on=['a'])

print(df.head())
df = cudf.merge(df1, df2, on=['a', 'b'])

print(df.head())
df = cudf.DataFrame({'a': np.repeat([0, 1, 2, 3], 25).astype(np.int32), 

                     'b': np.random.randint(2, size=100).astype(np.int32), 

                     'c': np.arange(0, 100).astype(np.int32), 

                     'd': np.arange(100, 0, -1).astype(np.int32)})

print(df.head())
grouped_df = df.groupby('a')

print(grouped_df)
aggregation = grouped_df.sum()

print(aggregation)
aggregation = df.groupby(['a', 'b']).sum()

print(aggregation)
categories = [0, 1, 2, 3]

df = cudf.DataFrame({'a': np.repeat(categories, 25).astype(np.int32), 

                     'b': np.arange(0, 100).astype(np.int32), 

                     'c': np.arange(100, 0, -1).astype(np.int32)})

print(df.head())
result = df.one_hot_encoding('a', prefix='a_', cats=categories)

print(result.head())

print(result.tail())
result = df.one_hot_encoding('a', prefix='a_', cats=[0, 1, 2])

print(result.head())

print(result.tail())