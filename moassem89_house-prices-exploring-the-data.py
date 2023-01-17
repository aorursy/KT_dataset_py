# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



home_data = pd.read_csv("../input/train.csv")

home_test = pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
#Discovering the data

print(home_data.columns)



home_data.describe()



home_data.head()
#Discovering the testing data

#As we can see, we have 38 numeric columns out of 81 columns in total

print(home_test.columns)



print(home_test.head())



home_test.describe()
#Note that the test data doesn't have the sale price

data_numeric = home_data.select_dtypes(include='number')

data_notnumeric =  home_data.select_dtypes(include='object')

print('Numeric features are:')

print(data_numeric.columns)

print('Categorical features are')

print(data_notnumeric.columns)

#For curiosity 

data_notnumeric.describe()
#Discovering the number of distinct values in caegorical columns

#data_notnumeric.apply(pd.Series.nunique)

state_counts = data_notnumeric.apply(lambda x: len(x.unique()))
state_counts.sort_values(ascending=False)
#Using a dictionary to list the different values of each columns

d = {}

for x in data_notnumeric.columns:

    d[x] = data_notnumeric[x].unique()
d