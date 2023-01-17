#Using Imputation to clean up the data

#make copy to aviod changing original data (when Imputing)
new_train_data = train_df.copy()
new_test_data = test_df.copy()
# make new columns?
one_hot_encoded_training_predictors = pd.get_dummies(new_train_data)
col_with_missing = (col for col in new_train_data["Age"]
                                if new_train_data["Age"].isnull().any())
for col in col_with_missing:
    new_train_data["Age"] = new_train_data["Age"].isnull()
# Imputation
my_imputer = Imputer()
new_train_data = my_imputer.fit_transform(new_train_data)
# create new dataframe with the dropped columns
train_df.drop(['PassengerId','Ticket','Fare','Cabin'], axis=1)
test_df.drop(['PassengerId','Ticket','Fare','Cabin'], axis=1)
train_df.isnull().sum()
p0 = (445-233) / 445* 100
print("Parch 0 = "+str(p0))
pd.crosstab(train_df["Parch"],train_df["Survived"])
#Subplot grid for plotting conditional relationships
es = sns.FacetGrid(train_df,
                 col="Survived",
                 row="Sex")
es = es.map(plt.hist,"Embarked")
#plot on class
ec = sns.FacetGrid(train_df,
                 col="Pclass",
                 row='Survived')
ec = ec.map(plt.hist,"Embarked")


#create new feature for name called title use regular expression to grab only titles in the Name feature e.g. Mr. regex to get title. r'\w+[A-Z]+\.'
#This code is the code that extracts from the feature 'Name' as a string using ([A-Za-z]+)\.
#Crosstab is used to identify the title to the sex
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
#Subplot grid for plotting conditional relationships
g = sns.FacetGrid(train_df,
                 col="Survived",
                 row='Sex')
g = g.map(plt.hist,"Age")

#draw barcharts
p = sns.FacetGrid(train_df,
                 col="Survived",
                 row='Sex')
p = p.map(plt.hist,"Pclass")

#Draw a categorical plot onto a Facetgrid
sns.factorplot(x="Pclass",
              y="Survived",
              hue="Sex",
              data=train_df)
train_df.isnull().sum()
train_df.describe()
train_df.head(20)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

# explore the training dataset by columns
print(train_df.info())
print('_'*40)
print(test_df.info())
