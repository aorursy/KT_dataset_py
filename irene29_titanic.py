# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# checking versions of useful libraries and loading necessary packages

import sys #access to system parameters

print("python version: {}".format(sys.version))



import pandas as pd #data processing and analysis

print("panda version: {}".format(pd.__version__))



import numpy as np #scientific computing package

print("numpy version: {}".format(np.__version__))



import scipy as sp #scientific computing and advanced mathematics

print("scipy version: {}".format(sp.__version__))



import matplotlib #scientific visualization

print("matplotlib version: {}".format(matplotlib.__version__))



import IPython 

from IPython import display #nice display of dataframes in jupyter notebook

print("IPython version: {}".format(IPython.__version__))



import sklearn #collection of machine learning algorithms

print("sklearn version: {}".format(sklearn.__version__))



# misc

import random

import time



# ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('_'*25)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading model libraries

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



# model helpers

from sklearn.preprocessing import OneHotEncoder , LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# visualization

from matplotlib import pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



# visualization defaults

%matplotlib inline

matplotlib.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
# meet and greet data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# copy the data

train_df_copy = train_df.copy(deep = True) #copy function creates new variable, (= in python pass value by reference)

combined_data = [train_df, test_df]



# basic info about data

train_df.info()



print('_'*25)



test_df.info()

train_df.sample(10)
# which features are present in the data

train_df.columns.values
# Four C's of cleaning data

# correcting: aberrations, outliers

# completing: missing values

# creating: new features

# converting: data formats
# correction: not required
# completion

# summarize missing values in train data



print("Train data column with total null values:\n",train_df.isnull().sum())



# conclusion: Age has 177 null values, Cabin has 687 null values, Embarked has 2 null values (out of 891)
# completion

# summarize missing values in test data



print("Test data column with total null values:\n",test_df.isnull().sum())



# conclusion: Age has 86 null values, Cabin has 327 null values, Embarked has 0 null values (out of 418)
# summary statistics of the numerical data

train_df.describe()
# Complete missing values 

# mean summarizes data well when data is not skewed

# Age: median (quantitative variable)

# Fare : median (quantitative variable)

# Embarked : mode (categorical variable)



for dataset in combined_data:

    # missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    

    # missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

    

    # missing embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)



    

# drop cabin, ticket, id

train_df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)



print(train_df.isnull().sum())

print('_'*25)

print(test_df.isnull().sum())

    
#test_df.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

test_df.isnull().sum()
# creating features

# 1. adding sibSp and parch to create a new feature - familysize

# 2. isalone feature from familysize

# 3. adding a title feature extracted from name

# 4. creating fareband from fare

# 5. creating ageband from age
# 1. adding sibSp and parch to create a new feature - familysize



for dataset in combined_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

train_df.sample(3)    
# 2. isalone feature from familysize



for dataset in combined_data:

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
# 3. adding a title feature extracted from name



for dataset in combined_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    

# different titles present in data

train_df['Title'].unique()



# We dont need to keep all the titles

# if number is greater than 10, we will keep the title, otherwise replace with misc

title_names = train_df['Title'].value_counts() < 10



# apply to each title value a function such that it replaces value with 'Misc' if value count is less that 10

train_df['Title'] = train_df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



# shows there are now only 5 different titles

train_df['Title'].value_counts()

# 4. creating fareband from fare

for dataset in combined_data:

    dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)

    

train_df.sample(3)    
# 5. creating ageband from age

for dataset in combined_data:

    dataset['AgeBand'] = pd.cut(dataset['Age'].astype(int), 5)

    

train_df.sample(10)     
# analyze relation between gender and survival rate.

# Explanatory variable : gender ( categorical )

# Response variable : survived ( categorical )

# C -> C relation is best examined by two way tables summarized by condional percentages

# we used mean because numebr of females and males might not be same. so to compare, we calculate the conditional percentage

# for each category of explanatory variable.

# in this case, how many females survived and how many males survived out of total number of females and males respectively

# 74 percent female survived, 18 percent male survived

# this verifies one of our assumptions



train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
# analyze relation between pclass and survival rate

# C->C

# 62 percent passengers survived from Pclass 1

# this also confirms one of the assumptions



train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean()
# analyze relation between between SibSp and survived

train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Parch and Survived

train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Parch and SibSp individually doesnt have any proper relation with Survived. We can may be  create new feature by combining these two.
# verify assumtions based on visualizations

# analyze relation between numerical data and survival

# numerical data like age is ratio ( highly precise ).  It can be converted to ordinal data (bands) which is less precise.

# with bands on x-axis and survival on y axis of histogram, we can analyze the relation



# initialized facet grid object. Survived categoried variabl takes two values - 0, 1

g = sns.FacetGrid(train_df, col='Survived')



# visualizing data on facetgrid

g.map(plt.hist, 'Age', bins=20)
# age has a correlation with survival. 

# most of the passengers are in middle range

# oldest passengers and infants mostly survived

#Actions:

# complete the missing values for age

# Convert age to bands
# correlating categorical features (embarked, pclass, sex) with solution goal (survived)

# row of facetgrid : embarked, pointplot : pclass (row), survived (col), sex (hue)

# visualising relation between embarked (categorical variable) and survived (categorical variable)

# there are three different values of embarked



#decisions: embarked and sex added to model training



grid = sns.FacetGrid(train_df, col='Embarked')

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()

# visualizing fare ( numerical ) with survial (categorical), keep embarked as constant, and for each gender separately

# people with high fare has more survival rate

# decisions: fare into bands (range)

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived')

grid.map(sns.barplot, 'Sex', 'Fare')

grid.add_legend()
# Now we have collected decisions based on data visualization and numerical summaries of data

# Now, we will execute decisions of correcting features, creating features, completing features

# Correcting features
