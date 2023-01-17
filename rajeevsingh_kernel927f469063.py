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
# Use pandas for file parsing, and for constructing data frames, which are matrices (rows/columns) of organized data

import pandas as pd

train = pd.read_csv('../input/train.csv')
# how many rows and columns are in our training set?

train.shape
# are there any gaps in our data?

train.info()
# look at the top 5 rows of the training data

train.head(5)
# matplotlib and seaborn are both data visualization tools. 

# matplotlib provides basic data plotting
import matplotlib.pyplot as plt

# seaborn provides the ability to generate more sophisticated graphs
import seaborn as sns
def survivalChart(dataset, category):
    survived = dataset[dataset['Survived']==1]
    survivedInCategory = survived[category].value_counts()
    dead = dataset[dataset['Survived']==0]
    deadInCategory = dead[category].value_counts()
    df = pd.DataFrame([survivedInCategory, deadInCategory])
    df.index = ['Survived','Dead']
    print(df.plot(kind='bar',stacked=True, figsize=(10,5), title=category))

category_variables = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']

for category in category_variables:
    survivalChart(train, category)
# extract titles from the otherwise noisy "Name" column

titles = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
titles.value_counts()
temp = train.copy()
temp['Title'] = titles

survivalChart(temp, 'Title')
title_mapping = {
    'Mr': 0,
    'Miss': 1,
    'Mrs': 1,
    'Master': 2,
    'Rev': 3,
    'Dr': 3,
    'Mlle': 4,
    'Col': 5,
    'Major': 5,
    'Mme': 4,
    'Sir': 2,
    'Countess': 4,
    'Ms': 4,
    'Don': 2,
    'Capt': 2,
    'Lady': 4,
    'Jonkheer': 2
}

curated_data = train.copy().drop(['Name'], axis=1)
curated_data['Title'] = temp['Title'].map(title_mapping)
survivalChart(curated_data, 'Title')
curated_data.head(15)
sex_mapping = {"male": 0, "female": 1}
curated_data['Sex'] = train['Sex'].map(sex_mapping)

survivalChart(curated_data, 'Sex')
# fill missing age with median age for each title
curated_data["Age"].fillna(curated_data.groupby("Title")["Age"].transform("median"), inplace=True)
def survivalFacet(dataset, category, floor=0, ceil=False):
    facet = sns.FacetGrid(dataset, hue="Survived",aspect=4)
    facet.map(sns.kdeplot, category, shade= True)
    facet.set(xlim=(floor, ceil if ceil else dataset[category].max()))
    facet.add_legend()
    plt.show() 
    
survivalFacet(curated_data, 'Age')
curated_data.loc[ train['Age'] <= 16, 'Age'] = 0,
curated_data.loc[(train['Age'] > 16) & (train['Age'] <= 25), 'Age'] = 1,
curated_data.loc[(train['Age'] > 25) & (train['Age'] <= 35), 'Age'] = 2,
curated_data.loc[(train['Age'] > 35) & (train['Age'] <= 60), 'Age'] = 3,
curated_data.loc[ train['Age'] > 60, 'Age'] = 4

curated_data.head()
survivalChart(curated_data, 'Age')
survivalFacet(train, 'Fare', 0, 100)
curated_data.loc[ train['Fare'] <= 30, 'Fare'] = 0 
curated_data.loc[ train['Fare'] > 30, 'Fare'] =1
curated_data.head(20)
train.Cabin.value_counts()
curated_data["has_cabin"] = train['Cabin'].isnull()
curated_data = curated_data.drop(['Cabin'], axis=1)
curated_data.head(20)
curated_data["FamilySize"] = train["SibSp"] + train["Parch"] + 1
curated_data = curated_data.drop(['SibSp', 'Parch'], axis=1)
survivalChart(curated_data, 'FamilySize')
curated_data.head()
curated_data = curated_data.drop(['Survived', 'Ticket', 'PassengerId', 'Embarked'], axis=1)
curated_data.head()
# Importing Classifier Modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

import numpy as np

target = train['Survived']
model = DecisionTreeClassifier()
score = cross_val_score(model, curated_data, target, cv=10, scoring='accuracy')
print(score)
print(round(np.mean(score)*100, 2))
model = RandomForestClassifier(n_estimators=13)
score = cross_val_score(model, curated_data, target, cv=10, scoring='accuracy')
print(score)
print(round(np.mean(score)*100, 2))
clf = SVC()
score = cross_val_score(model, curated_data, target, cv=10, scoring='accuracy')
print(score)
print(round(np.mean(score)*100, 2))