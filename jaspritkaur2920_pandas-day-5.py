# import library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import data

df = pd.read_csv('/kaggle/input/stocknews/upload_DJIA_table.csv')
# look at the top 5 rows in the data

df.head()
# let's drop date column by syntax 1

df.drop(['Date'], axis = 1)
# let's drop open column by syntax 2

df.drop(columns = ['Open'])
# check for the condition if Dataframe

df.head()
df.drop([0, 1])
# let's make a multiIndex Dataframe

midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'],

                             ['speed', 'weight', 'length']],

                     codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],

                            [0, 1, 2, 0, 1, 2, 0, 1, 2]])

df1 = pd.DataFrame(index=midx, columns=['big', 'small'],

                  data=[[45, 30], [200, 100], [1.5, 1], [30, 20],

                        [250, 150], [1.5, 0.8], [320, 250],

                        [1, 0.8], [0.3, 0.2]])

df1
# to drop the whole 'cow' row and 'small' column

df1.drop(index='cow', columns='small')
# to drop the subindex 'length'

df1.drop(index='length', level=1)
df2 = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],

                   "toy": [np.nan, 'Batmobile', 'Bullwhip'],

                   "born": [pd.NaT, "1940-04-25",

                            pd.NaT]})

df2
df2.dropna()
df2.dropna(axis = 'columns')
df2.dropna(how = 'all')
df2.dropna(thresh = 2)
df2.dropna(subset=['name', 'born'])
df2.dropna(inplace = True)

df2
# let's see the datatype of each column

df.dtypes
d = {'col1': [1, 2], 'col2': [3, 4]}

df3 = pd.DataFrame(data=d)

df3.dtypes
df3.astype('int32').dtypes
df3.astype({'col1': 'float'}).dtypes
def method1(x):

    return x * 2

df.Open.apply(method1).head(5)
df.Open.apply(lambda x : x * 2).head()