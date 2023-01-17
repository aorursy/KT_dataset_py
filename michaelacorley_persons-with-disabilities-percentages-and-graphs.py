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
data = pd.read_csv("/kaggle/input/disability-statistics-united-states-2018/By-Race-Disability.csv", header=0, index_col=0)

df_By_Race_Disability = pd.DataFrame(data, columns=data.columns, index=data.index)



data = pd.read_csv("/kaggle/input/disability-statistics-united-states-2018/By-Race-Total.csv", header=0, index_col=0)

df_By_Race_Total = pd.DataFrame(data, columns=data.columns, index=data.index)



data = pd.read_csv("/kaggle/input/disability-statistics-united-states-2018/Male-Female-Disability.csv", header=0, index_col=0)

df_Male_Female_Disability = pd.DataFrame(data, columns=data.columns, index=data.index)



data = pd.read_csv("/kaggle/input/disability-statistics-united-states-2018/Male-Female-Total.csv", header=0, index_col=0)

df_Male_Female_Total = pd.DataFrame(data, columns=data.columns, index=data.index)



df_By_Race_Disability = df_By_Race_Disability.applymap(lambda x: x.replace(',', ''))

df_By_Race_Total = df_By_Race_Total.applymap(lambda x: x.replace(',', ''))

df_Male_Female_Disability = df_Male_Female_Disability.applymap(lambda x: x.replace(',', ''))

df_Male_Female_Total = df_Male_Female_Total.applymap(lambda x: x.replace(',', ''))
a = pd.DataFrame.to_numpy(df_By_Race_Disability, dtype=None, copy=False)

b = pd.DataFrame.to_numpy(df_By_Race_Total, dtype=None, copy=False)

a = a.astype('float64')

b = b.astype('float64')

c=a/b



df_By_Race_Percent = pd.DataFrame(c, columns=df_By_Race_Total.columns.values, index = df_By_Race_Total.index.values)

df_By_Race_Percent
ax = df_By_Race_Percent.plot.bar(y='Total', rot=90)
a = pd.DataFrame.to_numpy(df_Male_Female_Disability, dtype=None, copy=False)

b = pd.DataFrame.to_numpy(df_Male_Female_Total, dtype=None, copy=False)

a = a.astype('float64')

b = b.astype('float64')

c=a/b



df_Male_Female_Percent = pd.DataFrame(c, columns=df_Male_Female_Total.columns.values, index = df_Male_Female_Total.index.values)

df_Male_Female_Percent
ax = df_Male_Female_Percent.plot.bar(y='Total', rot=0)