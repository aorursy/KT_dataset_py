# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

data.head()
data.info() #find out knowledge about data frame
data.columns# view columns
data.shape# number of columns and rows
print(data.Genre.value_counts(dropna=True))
data.describe() #find out data frame's count,max,min,median etc.
data.boxplot(column='Global_Sales',by='Year')

plt.show()
meltli=pd.melt(frame=dataN, id_vars='Name',value_vars=['Other_Sales','Global_Sales']) #merge other sales and global sales in one column

meltli


dataconc=pd.concat([data.head(),data.tail()], axis=0,ignore_index=True) #merge only first 5 values and last 5 values in one data frames

dataconc
datta=data['Global_Sales']

dataa=data['Other_Sales']

dataco=pd.concat([datta,dataa], axis=1)# merge only global sales and other sales in one data frame

dataco
data.dtypes # find out data types

data['Year'] = data['Year'].fillna(9999) #change NaN year values as 9999

data['Year']=data['Year'].astype('int') # change year's type as int

data.dtypes

data.info()
data.info() #find out knowledge about data frame
data.Publisher.value_counts(dropna=True)

#data.Publisher.tail(20)
data['Publisher'].fillna('Unknowing',inplace=True)

#data.Publisher.dropna(inplace=False)
assert  data.Publisher.notnull().all() #checking does exist Nan values or not, if return nothing,  
data.info()