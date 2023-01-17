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
data = pd.read_csv('../input/Pokemon.csv')

data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['Type 1'].value_counts(dropna = False))
data.describe()
data.boxplot(column = 'Attack',by = 'Legendary')
data_new = data.head()

data_new
#Tidy Data

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars=['Attack','Defence'])

melted
#Pivoting Data

melted.pivot(index = 'Name', columns = 'variable',values='value')
#Concentraring Data

data1 = data.head()

data2 = data.tail()

concentenated_data = pd.concat([data1,data2],axis = 0,ignore_index = True)

concentenated_data
data1 = data['Attack'].head()

data2 = data['Defense'].head()

con_dat_col = pd.concat([data1,data2],axis=1)# axis = 0: dataframes in row

con_dat_col
#Data Types

data.dtypes
#lets convert data types

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')

data['Attack'] = data['Attack'].astype('float')
data.dtypes
#MISSING DATA and TESTING WITH ASSERT

data.info()
data["Type 2"].value_counts(dropna = False)
# Lets drop nan values

data1 = data

data1["Type 2"].dropna(inplace = True)
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true

assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values

data["Type 2"].fillna('empty',inplace = True)
assert data['Type 2'].notnull().all()