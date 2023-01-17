# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/suicide_stats.csv')
data.head()
data.dtypes
# We can check how statistical description of our data
# however it is important to note the describe function retuns data of type float and int only
data.describe()
data.dtypes
#
data['age']
data.dtypes
data['age'].value_counts()
# visualize age values to check their distribution
data['age'].value_counts().plot(kind='bar')
plt.title('Age distribution')
plt.xlabel('Age Bracket')
plt.ylabel('Frequency')
#check for categorical viability of sex type
#we convert sex into categorical variable
data['sex'].value_counts()
data['sex'] = data['sex'].astype('category')
data.dtypes
#We need to check our data for null values to help us dtermine what to do the null values
data.isnull().any()

# In this case we have alot of null values for age, suicide_no and population.
data.isnull().sum()
# we need to fill the missing values in our dataset
# the best to do this is by filling with zeros (0) for suicides_no and mean for population 
data['suicides_no'] = data['suicides_no'].fillna(0)
data.isnull().sum()
# as observed the missing values in the suicide_no have been filled with zero hence wo do not have any missing values
data['suicides_no'].value_counts()
data.suicides_no.hist()
plt.title('Suicides')
plt.xlabel('Number of Suicides')
plt.ylabel('Frequency')
data['population'] = data['population'].fillna(data['population'].mean())
data.isnull().sum()
data['population'].value_counts()
data['population'].hist()
plt.title('Population')
plt.xlabel('Population distribution')
plt.ylabel('Frequency')
#The visualization shows that we have an even distribution for sex across the age brackets
pd.crosstab(data.age, data.sex).plot(kind='bar')
pd.crosstab(data['sex'], data['age']).plot(kind ='line')
table=pd.crosstab(data['age'], data['sex'])
table.div(table.sum(1).astype(int), axis=0)
# However I personally dont like the age bracket, therefor I will opt to take the median for each case
data['age'].replace('15-24 years', 19.5, inplace = True)
data['age'].replace('25-34 years', 29.5, inplace = True)
data['age'].replace('35-54 years', 44.5, inplace = True)
data['age'].replace('5-14 years', 9.5, inplace = True)
data['age'].replace('55-74 years', 64.5, inplace = True)
data['age'].replace('75+ years', 75, inplace = True)
data.head()
