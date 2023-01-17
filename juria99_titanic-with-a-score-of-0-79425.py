# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data analysis and wrangling

# import pandas as pd

# import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier





import re

import sklearn

import xgboost as xgb



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)



from sklearn.model_selection import KFold
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

combine = [train_data,test_data]
print(train_data.isnull().sum())

print('-'* 20)

print(test_data.isnull().sum())
print(train_data.info())

print('-'* 40)

print(test_data.info())
print(train_data.describe()) # describe the features of numerical attributes

print('-'* 40)

print(test_data.describe())
print(train_data.describe(include = ['O'])) # describe the features of non-numerical attributes

print('-'* 40)

print(test_data.describe(include = ['O']))
train_data.Embarked.unique()
for dataset in combine:

    dataset.Sex = dataset.Sex.map({'female': 0,'male': 1})

    dataset.Embarked = dataset.Embarked.map({'S': 0,'C': 1,'Q': 2})
train_data.Sex.unique
train_data.describe()
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']

train = train_data.drop(drop_elements, axis = 1)

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
#sex

train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
#age

g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
train_data['HaveCabin'] = train_data['Cabin'].isnull().map({True: 0, False: 1})

test_data['HaveCabin'] = test_data['Cabin'].isnull().map({True: 0, False: 1})
train_data.HaveCabin.describe()
train_data[['HaveCabin','Survived']].groupby(['HaveCabin']).mean().plot.bar()
train_data.corr()
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data.Name.head()
#We combine these datasets to run certain operations on both datasets together, for example, using the same algorithm to fill the missing values.

combine = [train_data, test_data]

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_data['Title'], train_data['Sex']) #可以用这个预测年龄
dataset['Title'].unique()
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
print(combine[0].Title.describe())

print('-'* 40)

print(combine[1].Title.describe())
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)



train_data.head()
dataset.groupby(['Title']).size()
dataset.groupby(['Title']).Age.mean()
# Wrangle Data

train_data.drop(['Cabin','Name','PassengerId','Ticket'], axis=1,inplace=True)

test_data.drop(['Name','Cabin','Ticket'], axis=1,inplace=True)

combine = [train_data, test_data]
train_data.describe()
train_data.Embarked.fillna(0,inplace = True)
test_data.Fare.fillna(test_data.Fare.median(),inplace = True)
train_data.info()
test_data.info()
'''

guess_ages = np.zeros((2,3))

guess_ages



for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()

'''
guess_ages = np.zeros((5,3))

for dataset in combine:

    for i in range(0, 5):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Title'] == i+1) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()

            

            # if there is no such guest of the target combination, then we use the 'Title' itself to make a prediction

            if pd.isnull(age_guess):

                age_guess = dataset[dataset['Title'] == i+1].Age.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 5):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Title == i+1) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)
train_data.head()
train_data['AgeBand'] = pd.cut(train_data['Age'], 8)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 10, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 70), 'Age'] = 6

    dataset.loc[(dataset['Age'] > 70) & (dataset['Age'] <= 80), 'Age'] = 7

train_data.head()
train_data.drop(['AgeBand'],axis = 1,inplace = True)

combine = [train_data, test_data]

train_data.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1,inplace = True)

test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1, inplace = True)

combine = [train_data, test_data]
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)

train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_data = train_data.drop(['FareBand'], axis=1)

combine = [train_data, test_data]
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()

X_train.info()
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "HaveCabin","Title","IsAlone"]

# 存放不同参数取值，以及对应的精度，每一个元素都是一个三元组(a, b, c)

results = []

# 最小叶子结点的参数取值

sample_leaf_options = list(range(1, 100, 3))

# 决策树个数参数取值

n_estimators_options = list(range(100,300, 5))

groud_truth = train_data['Survived'][601:]

 

#找最好的那棵树

for leaf_size in sample_leaf_options:

    for n_estimators_size in n_estimators_options:

        alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)

        alg.fit(train_data[predictors][:600], train_data['Survived'][:600])

        predict = alg.predict(train_data[predictors][601:])

        # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度

        results.append((leaf_size, n_estimators_size, (groud_truth == predict).mean()))

        # 真实结果和预测结果进行比较，计算准确率

        print((groud_truth == predict).mean())

 

# 打印精度最大的那一个三元组

print(max(results, key=lambda x: x[2]))
model = RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=4, random_state=1)

model.fit(X_train, Y_train)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# from sklearn.ensemble import RandomForestClassifier



# y = train_data["Survived"]



# X = pd.get_dummies(X_train)

# X_test = pd.get_dummies(X_test)



# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")