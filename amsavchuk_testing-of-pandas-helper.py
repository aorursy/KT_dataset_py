import numpy as np 

import pandas as pd 

from pandas_helper import convert_to_pretty_df
df = pd.DataFrame()

df['bools'] = ['On', 'Off', 'On', 'Off']

df['datetimes'] = ['2019-01-01', None, '2018-02-02', '2017-03-03']

df['strs'] = ['Moscow', None, 'Chicago', 'Las Vegas']

df['ints'] = ['2', 2, 'str', '1e-9']

df['floats'] = ['2.1', 2.02, 'str', '1e-9']
df.head()
df.info()
new_df = convert_to_pretty_df(df, errors='filter_out')

new_df.head()
# Pandas uses the object dtype for storing strings.

# https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes

new_df.info()
df = pd.DataFrame()

df['bools'] = ['On', 'Off', 'On', 'Off']

df['datetimes'] = ['2019-01-01', pd.NaT, '03/03/2003', '03/03/2002']

df['strs'] = ['Moscow', 1, 'Chicago', 'Las Vegas']

df['ints'] = ['2', 2, 'str', '1e-9']

df['floats'] = ['2.1', 2.02, 'str', '1e-9']



new_df = convert_to_pretty_df(df, errors='filter_out', columns=['bools', 'datetimes', 'strs'], 

                              column_type={'bools': str}, 

                              column_datetime_params={'datetimes': {'format': '%d/%m/%Y'}})

print(new_df.head())
# Pandas uses the object dtype for storing strings.

# https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes

new_df.info()
new_df = convert_to_pretty_df(df, errors='raise')

new_df.head()