import numpy as np

import pandas as pd
# df.stack?
df_a = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))

print(df_a, '\n')



df_s = df_a.stack()

print(df_s, '\n')
# df.unstack?
print(df_s, '\n')



df_u = df_s.unstack()

print(df_u)
# cheese.melt
cheese = pd.DataFrame({'first': ['John', 'Mary'],

                       'last': ['Doe', 'Bo'],

                       'height': [5.5, 6.0],

                       'weight': [130, 150]})

cheese
cheese.melt(id_vars=['first', 'last'], 

            value_vars=['height', 'weight'])
# df.pivot?
df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],

                   'bar': ['A', 'B', 'C', 'A', 'B', 'C'],

                   'baz': [1, 2, 3, 4, 5, 6],

                   'zoo': ['x', 'y', 'z', 'q', 'w', 't']})

df.head()
df.pivot(index='foo', columns='bar', values='baz')
# df.pivot_table?
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],

                   "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],

                   "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],

                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],

                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

df
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum)