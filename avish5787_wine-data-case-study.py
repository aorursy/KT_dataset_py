import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import scipy.stats as st

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
#Importing red wine dataset

red = pd.read_csv('/kaggle/input/winequality-red.csv',encoding = 'latin1',sep = ';')

red.head()
#Creating a new column Color

red['Color']='Red'

red.head()
#Importing red wine dataset

white = pd.read_csv('/kaggle/input/winequality-white.csv',encoding = 'latin1',sep = ';')

white.head()
white.isna().sum()

red.isna().sum()
#Creating a new column Color

white['Color'] = 'White'

white.head()
#Info method for red dataset

red.info()
#Info method for white dataset

white.info()
#1 To see shape of dataset

red.shape

white.shape
#2 To specifically see the columns in red dataset

red.columns
#To specifically see the columns in red dataset

white.columns
#3 To find the missing values in red wine dataset

red.isna().sum()
#To find the missing values in red wine dataset

white.isna().sum()
#4 Too find the duplicats in white dataset

white.duplicated().sum()
#5 To find the number of unique values present in both the dataset

red.nunique()

white.nunique()
#6 To find the mean density of Red dataset

red.density.mean()
#Combining the red dataset and white dataset into WINE DATASET

wine = red.append(white,ignore_index=True)

wine.head()
#1.Converting into Lower case

wine.columns = map(str.lower,wine.columns)

wine.head()
#2 Converting white spaces with underscore(_)

wine.columns = wine.columns.str.replace(' ','_')

wine.head()
#1.

wine.hist(figsize = (10,10));
#Grouping quality and finding the mean

wine1 = wine.groupby(by='quality').mean()

wine1.head()
#Groupby quality and ph

wine2 = wine.groupby(by =['quality'],as_index = False)['ph'].mean()

wine2.head()
#1 Groupy by color and quality and finding the mean

wine3 = wine.groupby(by = 'color')['quality'].mean()

wine3.head()
#describe function

wine['ph'].describe()
#2ou can create a categorical variable from a quantitative variable by creating your own categories. pandas' cut function let's you "cut" data in groups.

#Using this, create a new column called acidity_levels with these categories

wine['acidity_levels'] = pd.cut(x=wine['ph'],bins=[2.720000,3.110000,3.210000,3.320000,4.010000],labels=['High','Moderatley High','Medium','Low',],include_lowest=True)
#Groupby acidity levels and quality and the find the mean

wine4 = wine.groupby(by = 'acidity_levels')['quality'].mean()

wine4.head()
#Describe function for Alcohol

wine['alcohol'].describe()
#Use of query method to find if alcohol less than median of alcohol i.e 10.300000

wine_low = wine.query('alcohol < 10.30000') 

wine_low.head()
#Use of query method to find if alcohol greater and equal to than median of alcohol i.e 10.300000

wine_high = wine.query('alcohol >= 10.30000') 

wine_low.head()
#Calculating the mean quality of wine_low 

wine_low['quality'].mean()
#Calculating the mean quality of wine_high

wine_high['quality'].mean()
#Calculating the median for wine dataset

wine.median()
#Using query method find if residual_sugar greater and equal to than median of residual_sugar i.e 3.00000

wine_more_residual = wine.query('residual_sugar >=3.00000')
#Using query method find if residual_sugar less than  median of residual_sugar i.e 3.00000

wine_less_residual = wine.query('residual_sugar < 3.00000')
#Finding the mean 

wine_more_residual['residual_sugar'].mean()
wine_less_residual['residual_sugar'].mean()
#Correlation of wine dataset

wine.corr()
#Calculating the heatmap for wine dataset

plt.figure(figsize = (20,10))

sns.heatmap(wine.corr(), annot=True)
#Calculating the pairplot for wine dataset

sns.pairplot(wine)
wine.describe()
sns.distplot( wine['quality'])
wine_heat = wine[wine['quality']==5]
plt.figure(figsize = (20,10))

sns.heatmap(wine_heat.corr(), annot=True)
sns.distplot(wine_heat['quality'])