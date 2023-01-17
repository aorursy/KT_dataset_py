# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_excel('/kaggle/input/world-bank-climate-change-data/climate-change-excel-4-6-mb-.xls')
data
df = data.T.iloc[6:]
df.drop(df.tail(1).index,inplace=True) 
df = df.where(df.applymap(

    lambda x: str(x).isdigit()

))
df = df.fillna(df.median())
df = df.T
df  = df.fillna(df.median())
df.mean().plot()
df = df.astype(float)
df = pd.DataFrame(df,dtype = float)
data.iloc[:,6:30] = df
df_last = pd.DataFrame()

for name in data['Series name'].unique():

    df_last = df_last.append(data.loc[data['Series name'] == name].iloc[:,6:30].mean(),ignore_index=True )

            
df_last
df_last['names'] = data['Series name'].unique()
df_last.drop('names',axis = 1).T.corr().columns = data['Series name'].unique()
df_last.drop('names',axis = 1).T.corr()
import seaborn as sns

import matplotlib.pyplot as plt
sns.heatmap(df_last.drop('names',axis = 1).T.corr())
the_list = list(data['Series name'].unique())
the_list
the_list.index('Under-five mortality rate (per 1,000)')
df_last.drop('names',axis = 1).T.iloc[:,4].plot()
df_last.drop('names',axis = 1).T.iloc[:,7].plot()
df_last.drop('names',axis = 1).T.iloc[:,8].plot()