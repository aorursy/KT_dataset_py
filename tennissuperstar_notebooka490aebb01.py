# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Read in your data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Data dimensions - the data has 188318 entries and 132 columns

train.shape
# What does the data look like? 

train.head()
# Questions: Do we have missing values? Negative values? Categorical/ numerical variables?



# Results: 

# We determine there are no missing values as all the counts are 188318

# There are no negative values as all min > 0

# Describe() only includes numerical variables 

# To include categorical variables use describe(include=['object'])

# This tells us there are 116 categorical variables and 14 numerical variables



print(train.describe())

print(train.describe(include = ['object']))
# Remove the id column - for a python dataframe you can remove columns in two ways

# The axis parameter in the first option specifies we are dropping a column

# The function iloc [:, 1:] says keep all rows and all columns from 1 on therefore excluding

# the id column at index 0



train = train.drop('id', axis=1)

# train = train.iloc[:, 1:]





# Split the data into numerical and categorical data

categorical_data = train.iloc[:, :116]

numerical_data = train.iloc[:, 116:]

print(categorical_data.columns)
# Loop through each column of categorical data and get the unique categories.

# We will create a bar plot for each.


