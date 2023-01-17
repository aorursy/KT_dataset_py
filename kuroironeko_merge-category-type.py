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
df0 = pd.DataFrame({'id' : [0, 1, 2], 'value':['a', 'b', 'c']})

df1 = pd.DataFrame({'id' : [1, 2, 3], 'value1':['d', 'e', 'f']})

df2 = pd.DataFrame({'id' : [1, 0, 2], 'value2':['g', 'h', 'i']})

df3 = pd.DataFrame({'id' : [3, 4, 5], 'value':['j', 'k', 'l']})
for df in [df0, df1, df2, df3]:

    df['id'] = df['id'].astype('category')



df0['value'] = df0['value'].astype('category')

df3['value'] = df3['value'].astype('category')
df0.info()
df_m0 = pd.merge(df0, df1, how='outer', on='id')
df_m0
df_m0.info()
df_m1 = pd.merge(df0, df2, how='outer', on='id')
df_m1
df_m1.info()
df_m2 = pd.merge(df0, df3, how='outer')
df_m2
df_m2.info()
df_m3 = pd.merge(df0, df0, how='outer')
df_m3
df_m3.info()