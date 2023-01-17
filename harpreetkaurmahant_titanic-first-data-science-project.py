# data analysis and wrangling



import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # Linear Algebra

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_df.columns
print('_'*40)

train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df[train_df['Embarked'].isnull()]
train_df[train_df['Age'].isnull()]
train_df[train_df['Cabin'].isnull()]
train_df[train_df['Embarked'].isnull()]
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df[train_df['Age'].isnull()]
train_df['Age'].describe()
for name in train_df['Name']:

    train_df['Title'] = train_df['Name'].str.extract('([A-Za-z]+)\.',expand=True)
pd.unique(train_df['Title'])
groupByTitle = train_df.groupby('Title').mean()



groupByTitle.head(20)
train_df[train_df['Age'].isnull()].groupby('Title', as_index=False).count()
#groupByTitle['Age']['Dr']



titlesWithMissingAge = ['Dr','Master','Miss','Mr','Mrs']



for title in titlesWithMissingAge:

    train_df.loc[train_df.Age.isnull() & (train_df.Title == title),'Age'] = groupByTitle['Age'][title]

train_df.isnull().sum()
test_df.info()
for name in test_df['Name']:

    test_df['Title'] = test_df['Name'].str.extract('([A-Za-z]+)\.',expand=True)
pd.unique(test_df['Title'])
groupByTestTitle = test_df.groupby('Title').mean()



groupByTestTitle
test_df[test_df['Age'].isnull()].groupby('Title', as_index=False).count()
title = ['Ms','Master','Miss','Mr','Mrs']



for title in test_df['Title']:

    test_df.loc[(test_df.Age.isnull()) & (test_df['Title'] == title),'Age'] = groupByTestTitle['Age'][title]

    

test_df.isnull().sum()
test_df[test_df['Age'].isnull()]
test_df.loc[test_df['Title']=='Ms']
test_df['Age'] = test_df['Age'].fillna(21)

test_df.isnull().sum()
test_df[test_df.Fare.isnull()]
test_df.groupby('Pclass')['Fare'].mean()
test_df['Fare'] = test_df['Fare'].fillna(12.4596)
test_df.isnull().sum()
women = train_df.loc[train_df.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

print('% of women who survived = ', rate_women)
men = train_df.loc[train_df.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print('% of men who survived = ', rate_men)
train_df.groupby(['Sex', 'Survived'] )['Survived'].count().unstack('Sex').plot(kind='bar')
train_df['Sex'] = train_df.Sex.apply(lambda x: 0 if x == "female" else 1)

test_df['Sex'] = test_df.Sex.apply(lambda x: 0 if x == "female" else 1)

train_df.drop('Cabin', axis=1)
test_df.drop ('Cabin', axis=1)
Emb_Keys={'C':1,'Q':2,'S':3}



train_df = train_df.replace({'Embarked':Emb_Keys})



test_df = test_df.replace({'Embarked':Emb_Keys})
Keys = {'Capt': 1, 'Col': 1, 'Don':1, 'Dr':1, 'Jonkh eer':1, 'Major':1, 'Rev': 1, 'Sir':1,'Mr':1, 'Countess': 2, 'Dona':2, 'Lady':2,'Mme':2, 'Mrs':2, 'Ms': 3, 'Mlle':3, 'Miss':3,'Master':4}



# Remap the values of the dataframe 

train_df = train_df.replace({'Title':Keys})

test_df = test_df.replace({'Title':Keys})



for age in train_df['Age']:

    

    train_df.loc[(train_df['Age'] < 18),'Is_child'] = 1

    train_df.loc[(train_df['Age'] >= 18),'Is_child'] = 0



train_df.sample(5)
for age in test_df['Age']:

    

    test_df.loc[(test_df['Age'] < 18),'Is_child'] = 1

    test_df.loc[(test_df['Age'] >= 18),'Is_child'] = 0



test_df.loc[test_df['Is_child'] == 1].sample(5)
train_df['Is_Alone'] = train_df['SibSp'] + train_df['Parch']

train_df['Is_Alone'] = train_df.Is_Alone.apply(lambda x:1 if x == 0 else 0)



test_df['Is_Alone'] = test_df['SibSp'] + test_df['Parch']

test_df['Is_Alone'] = test_df.Is_Alone.apply(lambda x:1 if x == 0 else 0)
train_df['Family_size'] = train_df['SibSp'] + train_df['Parch'] + 1

test_df['Family_size'] = test_df['SibSp'] + test_df['Parch'] + 1
train_df['Individual_fare'] = train_df['Fare'] / train_df['Family_size']

test_df['Individual_fare'] = test_df['Fare'] / test_df['Family_size'] 
train_df.sample(5)
test_df.sample(5)
plt.subplots(figsize = (15,10))

sns.heatmap(train_df.drop(columns='PassengerId').corr(), annot=True,cmap="RdYlGn_r")

plt.title("Feature Correlations", fontsize = 18)
from sklearn.ensemble import RandomForestClassifier



y = train_df["Survived"]



#Select features

features = ["Pclass", "Sex", "Is_Alone", "Family_size", "SibSp", "Parch"]



X = pd.get_dummies(train_df[features])

X_test = pd.get_dummies(test_df[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

#model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=7)

model.fit(X, y)

predictions = model.predict(X_test)





output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
output.groupby('Survived').count()
output.head()
test_df.head(15)