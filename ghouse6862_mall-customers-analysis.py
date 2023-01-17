# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# reading in the csv file using pandas.

df = pd.read_csv('../input/Mall_Customers.csv')
# printing the first 5 rows.

df.head()
# checking for null values.

df.isnull().sum()
#import libraries to perform visualization.

import matplotlib.pyplot as plt

import seaborn as sns
#You'll need to use this line to see plots in the notebook.

%matplotlib inline
#Set the aesthetic style of the plots.

sns.set_style('whitegrid')
# using box plot to see the distribution of Annual Income (k$) with respect to Gender.

sns.boxplot(x='Gender',y='Annual Income (k$)',data=df,palette='rainbow')
# using box plot to see the distribution of Spending Score (1-100) with respect to Gender.

sns.boxplot(x='Gender',y='Spending Score (1-100)',data=df,palette='coolwarm')

# from the plot it looks like females tend to spend higher.
# barplot is a general plot that allows you to aggregate the categorical data based off some function, by default the mean.

sns.barplot(x='Gender',y='Spending Score (1-100)',data=df)

# from this plot also it looks like females tend to spend higher.
# lets check the relationship between Age and Annual Income.

sns.jointplot(x='Age',y='Annual Income (k$)',data=df)

# people in age group 30-50 are likely to earn more.
# lets check the relationship between Annual Income and Spending Score.

sns.jointplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=df)
# lets check the relationship between Age and Spending Score.

sns.jointplot(x='Age',y='Spending Score (1-100)',data=df)

# this plot shows that people whose age is more than 40 tend to spend less.
# import libraries to perform customer segmentation.

import sklearn

from sklearn.cluster import KMeans
# drop CustomerID since it is of no use in training model.

df.drop(columns='CustomerID',inplace=True)
# converting categorical features to dummy variables, as our machine learning algorithm won't be able to directly take in this features as inputs.

sex = pd.get_dummies(df['Gender'],drop_first=True)

df.drop(columns='Gender',inplace=True)

df['Sex'] = sex
# printing first 5 rows of converted dataframe.

df.head()
# create the model, specifying the number of clusters needed.

kmeans = KMeans(n_clusters=2)
# fit model to all the data.

kmeans.fit(df)
# storing labels in a variable.

clusters = kmeans.labels_
# add labels to the dataframe.

df['group'] = clusters
df.head()
# visualizing clusters using lmplot. This plots will help us to get our target customers.

sns.lmplot(x='Age',y='Spending Score (1-100)',data=df,hue='group')