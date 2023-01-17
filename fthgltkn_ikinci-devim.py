# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/Pokemon.csv')

df=pd.DataFrame(data)

data=data.set_index("#")

data.columns
data.head()
data.tail()
data.describe()
data.info()
data['Type 1'].value_counts(dropna =False)
data['Type 2'].value_counts(dropna=False)
data['Type 2'].fillna('empty',inplace=True)

assert 1==1
data['Type 2'].value_counts(dropna=False)
data.boxplot(column='Attack',by='Legendary')
data_new=data.head()
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1=data.head()

data2=data.tail()

concet_row_data=pd.concat([data1,data2],axis=0)

concet_row_data
data1=data['Attack'].head()

data2=data['Defense'].head()

concat_col_data=pd.concat([data1,data2],axis=1)

concat_col_data