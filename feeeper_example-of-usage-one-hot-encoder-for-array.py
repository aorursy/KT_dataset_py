import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from one_hot_encoder_for_array import one_hot_array_values
df = pd.DataFrame({'tags': ['tag1, tag2', 'tag2, tag3', 'tag1, tag4']})

df2 = one_hot_array_values(df['tags'])

df.join(df2)
df = pd.DataFrame({'tags': ['tag1, tag2', 'tag2, tag3', np.nan]})

df2 = one_hot_array_values(df['tags'], fillna_value='None')

df.join(df2)
df = pd.DataFrame({'tags': ['tag1; tag2', 'tag2; tag3', 'tag1; tag4']})

df2 = one_hot_array_values(df['tags'], fillna_value='None', sep=';')

df.join(df2)
df = pd.DataFrame({'tags': [1, 2, 3]})

df2 = one_hot_array_values(df['tags'], fillna_value='None', sep=';')

df2
df = pd.DataFrame({'tags': ['tag1; tag2', 'tag2; tag3', 'tag1, tag4']})

df2 = one_hot_array_values(df['tags'], sep=';')

df.join(df2)
df = pd.DataFrame({'tags': ['tag1; tag2', 'tag2; tag3', '']})

df2 = one_hot_array_values(df['tags'], fillna_value='None', sep=';')

df.join(df2)
df = pd.DataFrame({'tags': [['tag1', 'tag2'], ['tag2', 'tag3'], ['tag1', 'tag4']]})

df2 = one_hot_array_values(df['tags'], with_brackets=True)

df.join(df2)
df = pd.DataFrame({'tags': [[1, 2], [2, 3], [3, 5]]})

df2 = one_hot_array_values(df['tags'], with_brackets=True)

df.join(df2)