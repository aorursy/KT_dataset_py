# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.DataFrame({'start_year':[2000, 2001, 2002],

                  'end_year':[2010, 2011, 2012],

                  'price':[1.0, 2.0, 3.0]})
df
df_p = df.pivot('start_year', 'end_year', 'price')

df_p
# stack table is to change  column names to a level of index

df2 = df_p.stack()

df2
# see now index becomes MultiIndex

df2.index
# reset_index is to change all index into columns

df2.reset_index()
# name input is to put name on the value column

df2.reset_index(name='test')
df
# take max on the last column, return a boolean series

temp_df = df['price']==df['price'].max()

temp_df
# the whole row contains contains the max of the last column

df[temp_df]
# use method also works

df.loc[temp_df]
temp_df_2 = df['price']<3

temp_df_2
df[temp_df_2]
df
temp_df_3 = df.loc[2] == 2012

temp_df_3
df.loc[:, temp_df_3]
temp_df_4 = df.loc[2] > 4

temp_df_4
df.loc[:, temp_df_4]
temp_df_5 = (df.loc[2] > 4) & (df.loc[2]< 2003)

temp_df_5
df.loc[:, temp_df_5]