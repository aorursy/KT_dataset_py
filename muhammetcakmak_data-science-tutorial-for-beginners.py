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
# İmport the data 

df=pd.read_csv('../input/breastCancer.csv')
# we can learn general information about the data.

df.info()

# It shows that all features are integer while bare_nucleoli is object (string)

# There are not null number and it has 699 rows and 11 columns. We can also see columns names.
# Let's look at data quickly 

df.head(7) 
# Correlation 

df.corr()

# This shows the correlation between the features. This is very important for data scientist

# Because we dont want to use features which are highly correalted with each other. We need

# to determine the correlated features and delete some of them by using techniques.
# Correlation visualisation

# Heatmap is good method to visualize correlation between features.

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(df.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)
#Scatter plot is a useful method to see the correlation between 2 features

df.plot(kind='scatter', x='size_uniformity', y='shape_uniformity', alpha=.5, color='red' )

plt.xlabel('size_uniformity')

plt.ylabel('shape_uniformity')

plt.title('size_uniformity --- shape_uniformity scatter plot')
#Histogram is as good method to see the distribution of a data.It also get called as frequency

df['class'].plot(kind='hist',color='red')
#valu_counts gives the numbers of the unique data in the selected feature

df['class'].value_counts()
# Describe method allow us to have statistic values of the data

df.describe()
# Quartile shows the improper data in the selected feature.we need to make sure that they are true

df.boxplot('mitoses', 'class')
# concatenating gives chance to make new dataframe by using selected features

data1 = df['size_uniformity']

data2 = df['shape_uniformity']

data3 = df['class']

df1 = pd.concat([data1, data2, data3],axis=1, ignore_index=False)

df1.head()
# missing data must be detected and handled properly such as deleting and changing with othher

# proper data. one of useful method is to use dropna() and fillna().

df['bare_nucleoli'].value_counts(dropna=True) # this gives all data with NaN data.

# we have 16 ? data which are not number
# Changing the any values in the feature

# I want to change '?' data in bare_nucleoli feature because '?' is not a number

# I will write 29 instead of '?' numbers in bare_nucleoli feature. 

df['bare_nucleoli'].values[df['bare_nucleoli'] == '?'] = 29

df['bare_nucleoli'].value_counts(dropna=True)

# Dropping the missing values with dropna()

data_drop = df.size_uniformity.dropna(inplace=False) 

# Filling the missing values with anything by using fillna(). Here we will with 'empty'

data_drop1 = df['shape_uniformity'].fillna('empty', inplace=False) 

# Finding values in the certain location (index) by using loc method

df.loc[1,['size_uniformity']] # it gives the first row of the size_uniformity feature
# Finding values in the certain location (index) by using loc method

#This selects the first 5 rows of the features between size_uniformity and mitoses.

df.loc[1:5, 'size_uniformity':'mitoses']
df['shape_uniformity'].value_counts(dropna=True)
# Filtering the data

# choosing data if shape_uniformity is equal to 1 and size_uniformity is equal to 2

data3 = df[(df['shape_uniformity'] == 1) & (df['size_uniformity'] == 2)]

data3
# Defining the new feature.

# We can defining the new features by using the old features

df['new'] = df.size_uniformity + df.shape_uniformity

df.head()
# Let's see the new feature

df.head()
#Stacking the features

a=df['size_uniformity']

b=df['shape_uniformity']

np.vstack((a,b)) # it stacking two columns side by side

np.hstack((a,b)) # it stacking two columns one under the other

# Split the data into 2 groups as features and class

X_data = df.drop(["id","new","class"],axis=1,inplace=False)# features columns

Y_data = df["class"].values # class column 

Y_data