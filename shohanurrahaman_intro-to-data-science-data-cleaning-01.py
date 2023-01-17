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
df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')



print('missing value in every column :', df.isnull().sum())

print('total missing value of all the columns : ', df.isnull().sum().sum())



#show all the rows of missing value 

display(df[df['total_bedrooms'].isnull()])
print('shape of original dataframe : ', df.shape)



# remove missing rows 

new_df = df.dropna()

print('shape of filtered dataframe :', new_df.shape)



# remove the whole column (bad idea)

new_df = df.dropna(axis=1)

print('shape of filtered dataframe : ', new_df.shape)
print("Total missing values : ",df.isnull().sum().sum())



#fill value with median value 

value = df['total_bedrooms'].median()

new_df = df['total_bedrooms'].fillna(value)



#fill missing value with interpolation 

new_df = df.interpolate('linear')



print("Total missing values : ",new_df.isnull().sum().sum())



#print(new_df.loc[290, 'total_bedrooms'])