# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read data DC and Marvel

dc = pd.read_csv('../input/dc-wikia-data.csv')

dc.head(3)
mr = pd.read_csv('../input/marvel-wikia-data.csv')

mr.head(3)
# We can control whether data has NaN value or not, the code below 

# assert pd.notnull(df).all().all()
# Look at the info; data types, non-null entry numbers, range index and memory usage.

dc.info()
# We can see number of NaN values of the features:

dc.isna().sum()
# GSM feature is not usable for me, because there are only 64 values. So we can drop:

dc = dc.drop('GSM', axis = 1)

dc.head(3)
# String characters should be regulated. If there are any space character there will be problem.

for col in dc.select_dtypes([np.object]):

    dc[col] = dc[col].str.strip()
# We search for categorical data. 'ID', 'SEX', 'ALIVE' are categorical. We should change types of this features. 

# Memory usage will decrease and usability will increase.

dc['ID'].value_counts()
dc['ID'] = dc['ID'].astype('category')
dc['SEX'].value_counts()
dc['SEX'] = dc['SEX'].astype('category')
dc['ALIVE'].value_counts()
dc['ALIVE'] = dc['ALIVE'].astype('category')
# 'urlslug' feature has pattern: '\/wiki\/ *******_(****_****)' 

dc['urlslug'].head(5)
# We replace 'wiki' with ''

dc['urlslug'] = dc['urlslug'].str.replace('wiki', '')
# After that, we split the remaining pattern '\/\/ *******_(****_****)' with '(' and 'list1': ('\/\/ *******_', '****_****)' )  is created as new column.

dc['list1']= dc['urlslug'].str.split('(')
# 'urlslug' feature is dropped.

dc.drop('urlslug', axis=1)

dc.head(2)
# First element of 'list1' is assigned to 'urlslug1', second element is assigned to 'urlslug2' columns.

dc['urlslug1'] = dc['list1'].str.get(0)
dc['urlslug2'] = dc['list1'].str.get(1)
# 'list1' column is dropped

dc.drop('list1', axis=1,inplace=True)
dc.head(5)
# '_' character of 'urlslug1' and 'urlslug2' columns are replaced with ''.

dc['urlslug1'] = dc['urlslug1'].str.replace('_', ' ')
dc['urlslug2'] = dc['urlslug2'].str.replace('_', ' ')
# ')' character of 'urlslug2' column  is replaced with ''

dc['urlslug2'] = dc['urlslug2'].str.replace(')', '')
dc.head()

#'urlslug1' still contains  '\/\/' characters.
# First, with '\\', '\' character  is replaced with  ''.

dc['urlslug1'] = dc['urlslug1'].str.replace('\\', '')
# With '\/', '/' character  is replaced with  ''.

dc['urlslug1'] = dc['urlslug1'].str.replace('\/', '')
# We control whether there are space character ' ' in that column.

dc['urlslug1'].apply(len).head()
# For example, 'batman' has 6 characters, but we can see upward that length is 7, 'supermen' has 8 characters, but we can see upward that length is 7

# We should use .strip() method in order to drop space characters.

dc['urlslug1'] = dc['urlslug1'].str.strip()
# Could do it?

dc['urlslug1'].apply(len).head()

# Yes. length of 'batman' is 6, and so on..
dc.head()
# Drop 'urlslug':

dc.drop('urlslug', axis=1, inplace=True)
# Feature 'name' is the same with 'urlslug1' + 'urlslug2' so we can drop:

dc.drop('name', axis=1, inplace=True)
dc.head()
#'ALIGN', 'EYE' and 'HAIR' columns are also categorical data types.

dc['ALIGN'].value_counts()
dc['ALIGN'] = dc['ALIGN'].astype('category')
dc['EYE'].value_counts()
dc['EYE'] = dc['EYE'].astype('category')
dc['HAIR'].value_counts()
dc['HAIR'] = dc['HAIR'].astype('category')
# 'FIRST APPEARANCE' column includes year and month pair. Also, we have 'YEAR' column. 

# We split this column

dc['list2'] = dc['FIRST APPEARANCE'].str.split(',')
#'MONTH' column is created from the second element of 'list2'

dc['MONTH'] = dc['list2'].str.get(1)
dc['MONTH'].value_counts()
dc['MONTH'] = dc['MONTH'].str.replace('August','Aug')

dc['MONTH'] = dc['MONTH'].str.replace('December','Dec')

dc['MONTH'] = dc['MONTH'].str.replace('October','Oct')

dc['MONTH'] = dc['MONTH'].str.replace('September','Sep')

dc['MONTH'] = dc['MONTH'].str.replace('July','Jul')

dc['MONTH'] = dc['MONTH'].str.replace('February','Feb')

dc['MONTH'] = dc['MONTH'].str.replace('June','Jun')

dc['MONTH'] = dc['MONTH'].str.replace('March','Mar')

dc['MONTH'] = dc['MONTH'].str.replace('January','Jan')

dc['MONTH'] = dc['MONTH'].str.replace('April','Apr')

dc['MONTH'] = dc['MONTH'].str.replace('November','Nov')
# 'MONTH' is also categorical

dc['MONTH'] = dc['MONTH'].astype('category')
# No need to 'FIRST APPEARANCE' and 'list2' columns

dc.drop('FIRST APPEARANCE', axis=1, inplace=True)
dc.drop('list2', axis=1, inplace=True)
dc.head()
dc.info()
# There are only 3 numeric columns.

dc.describe()
# 'page_id' column is not needed.

dc.drop('page_id', axis= 1, inplace=True)
# There are NaN values, but data types are regulated, complex columns are cleaned.

dc.head()
mr.head()
# Same regulations are performed to marvel data.

mr.drop(labels= ['page_id', 'urlslug', 'GSM'], axis= 1, inplace=True)
mr['list1'] = mr['name'].str.split('(')
mr['name1'] = mr['list1'].str.get(0)
mr['name2'] = mr['list1'].str.get(1)
mr.drop(['name', 'list1'], axis = 1, inplace=True)
mr['name2'] = mr['name2'].str.replace(')', '')
mr['name2'] = mr['name2'].str.replace('\\', '')
mr['name2'] = mr['name2'].str.replace('"', '')
mr['name2'] = mr['name2'].str.replace('-', '')
mr['name1'] = mr['name1'].str.replace('-', '')
mr['list2'] = mr['FIRST APPEARANCE'].str.split('-')
mr['MONTH'] = mr['list2'].str.get(0)
mr.drop(['FIRST APPEARANCE', 'list2'], axis=1, inplace=True)
mr.head()
mr['ID'].value_counts()
mr['ID'] = mr['ID'].astype('category')
mr['ALIGN'].value_counts()
mr['ALIGN'] = mr['ALIGN'].astype('category')
mr['EYE'].value_counts()
mr['EYE'] = mr['EYE'].astype('category')
mr['HAIR'].value_counts()
mr['HAIR'] = mr['HAIR'].astype('category')
mr['SEX'].value_counts()
mr['SEX'] = mr['SEX'].astype('category')
mr['ALIVE'].value_counts()
mr['ALIVE'] = mr['ALIVE'].astype('category')
mr['MONTH'].value_counts()
mr['MONTH'] = mr['MONTH'].astype('category')
mr.head()
mr.head(2)
dc.head(2)
# Column names should be identical:

dc.rename(columns={'urlslug1': 'name1',

                   'urlslug2': 'name2',

                  'YEAR': 'Year'}, inplace=True)

dc.head(2)