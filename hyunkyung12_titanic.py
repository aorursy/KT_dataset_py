# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.cross_validation import KFold

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



# Any results you write to the current directory are saved as output.gender_submission.csv



input_file = ["../input/gender_submission.csv",

              "../input/train.csv",

              "../input/test.csv"]



gender = pd.read_csv(input_file[0], header = 0)

train = pd.read_csv(input_file[1], header = 0)

test = pd.read_csv(input_file[2], header = 0)

train.info()
test.info()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # setting seaborn default for plots
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts() # survived 라는 값에 대해 수를 세줌

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex') # 생존자 중 여성이 많음을 알 수 있다.
bar_chart('Pclass') # 3등석의 사람들이 많이 죽음을 알 수 있다.
bar_chart('SibSp') # 동승자 중 형제나 배우자가 없는 사람이 훨씬 많이 죽음을 알 수 있다.
bar_chart('Parch') # 동승자 중 부모님이나 자식이 없는 사람이 훨씬 더 많이 죽음을 알 수 있다
bar_chart('Embarked') # S 자리에 있던 사람들이 많이 죽었음을 알 수 있다.
train['Sex'].value_counts()
train_test_data = [train, test] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
#train.drop('Name', axis=1, inplace=True)

#test.drop('Name', axis=1, inplace=True)

#train.drop('Title', axis=1, inplace=True)

#test.drop('Title', axis=1, inplace=True)
train.info()
facet = sns.FacetGrid(train, aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.add_legend()

 

plt.show()
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
train.head(30)

train.groupby("Title")["Age"].transform("median")
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'editAge',shade= True)

facet.set(xlim=(0, train['editAge'].max()))

facet.add_legend()

 

plt.show()
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,

    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
bar_chart('Age')
train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = test['Embarked'].fillna('S')
embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

 

plt.show()
for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,

    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,

    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,

    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
cabin_mapping = {"A": 2.8, "B": 2.4, "C": 2, "D": 1.6, "E": 1.2, "F": 0.8, "G": 0.4, "T": 0}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade= True)

facet.set(xlim=(0, train['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
#train['FamilySize'] = list(map(lambda x: 1 if x >1 else 0, train['FamilySize']))
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
#test['FamilySize'] = list(map(lambda x: 1 if x >1 else 0, test['FamilySize']))
train['Sex'] = list(map(lambda x: 1 if x=='female' else 0, train['Sex']))
test['Sex'] = list(map(lambda x: 1 if x=='female' else 0, test['Sex']))
features_drop = ['Ticket', 'SibSp', 'Parch', 'PassengerId']

features_drop1 = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop1, axis=1)
train.info()

train = train.drop('Name',axis=1)

test = test.drop('Name',axis=1)
train_data = train.drop('Survived', axis=1)

target = train['Survived']

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import VotingClassifier

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

def classifier(clf):

    scoring = 'accuracy'

    score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

    print(round(np.mean(score)*100,2))
clf = [KNeighborsClassifier(n_neighbors = 13),

       DecisionTreeClassifier(),

       RandomForestClassifier(n_estimators=13),

       SVC(),

       XGBClassifier(),

       XGBClassifier(

             learning_rate =0.01,

             n_estimators=1000,

             max_depth=3,

             min_child_weight=3,

             gamma=0,

             subsample=0.7,

             colsample_bytree=0.7,

             nthread=3,

             scale_pos_weight=1,

             seed=27

       )]
result = []

for clfs in clf:

    acc = classifier(clfs)

    result.append(result)

print(result)
clf1 = KNeighborsClassifier(n_neighbors = 13)

clf2 = DecisionTreeClassifier()

clf3 = RandomForestClassifier(n_estimators=13)

clf4 = SVC()

clf5 = XGBClassifier()

clf6 = XGBClassifier(

             learning_rate =0.01,

             n_estimators=1000,

             max_depth=3,

             min_child_weight=3,

             gamma=0,

             subsample=0.7,

             colsample_bytree=0.7,

             nthread=3,

             scale_pos_weight=1,

             seed=27

       )

eclf = VotingClassifier(estimators=[('knn', clf1), 

                              ('dt', clf2), 

                              ('rf', clf3), 

                              ('svc', clf4), 

                              ('xgb', clf5), 

                              ('xgb1', clf6)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6 ,eclf], ['KNN','Decision Tree', 'Random Forest', 'SVM','xgboost','tuned_xgboost', 'Ensemble']):

    scores = cross_val_score(clf, train_data, target, cv=5, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
clf = XGBClassifier()

clf.fit(train_data, target)



test_data = test.drop("PassengerId", axis=1).copy()

prediction = clf.predict(test_data)


submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('submission.csv')

submission.head()