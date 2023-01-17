import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))
drinks = pd.read_csv('../input/drinks.csv')

pd.__version__
pd.show_versions()

#You can see the versions of Python, pandas, NumPy, matplotlib, and more.
df = pd.DataFrame({'col one':[100, 200], 'col two':[300, 400]})

df
pd.DataFrame(np.random.rand(4, 8))

#That's pretty good, but if you also want non-numeric column names, you can coerce a string of letters to a list and then pass that list to the columns parameter:
pd.DataFrame(np.random.rand(4, 8), columns=list('abcdefgh'))

#As you might guess, your string will need to have the same number of characters as there are columns.
df
df = df.rename({'col one':'col_one', 'col two':'col_two'}, axis='columns')

print(df)
df.columns = ['col_one', 'col_two']

df.columns = df.columns.str.replace(' ', '_')
df
df.add_prefix('X_')
df.add_suffix('_Y')
#Let's take a look at the drinks DataFrame:

drinks.head()
drinks.loc[::-1].head()
drinks.loc[::-1].reset_index(drop=True).head()

drinks.loc[:, ::-1].head()
#Here are the data types of the drinks DataFrame:

drinks.dtypes
drinks.select_dtypes(include='number').head()
drinks.select_dtypes(include='object').head()
drinks.select_dtypes(include=['number', 'object', 'category', 'datetime']).head()
drinks.select_dtypes(exclude='number').head()
df = pd.DataFrame({'col_one':['1.1', '2.2', '3.3'],

                   'col_two':['4.4', '5.5', '6.6'],

                   'col_three':['7.7', '8.8', '-']})

df
df.dtypes
df.astype({'col_one':'float', 'col_two':'float'}).dtypes
pd.to_numeric(df.col_three, errors='coerce')
pd.to_numeric(df.col_three, errors='coerce').fillna(0)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

df
df.dtypes