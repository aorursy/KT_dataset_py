# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

import cufflinks as cf

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

combine = [train, test]
pandas_profiling.ProfileReport(train)
print("Before", train.shape, test.shape, combine[0].shape, combine[1].shape)



train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)

combine = [train, test]



print("After", train.shape, test.shape, combine[0].shape, combine[1].shape)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])    
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',

                                                'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()    

train.head()
title_map = {"Master": 1, "Miss": 2, "Mr": 3, "Mrs": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_map)

    dataset['Title'] = dataset['Title'].fillna(0)

     

train.head()
train = train.drop(['Name', 'PassengerId'], axis=1)   

test = test.drop(['Name'], axis=1)

combine = [train, test]

train.shape, test.shape
sex_map = {"female": 0, "male": 1}

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map(sex_map)



train.head()    
guess_ages = np.zeros((2,3))



for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess = dataset[(dataset['Sex'] == i) & \

                           (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess.median()



    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

            

    dataset['Age'] = dataset['Age'].astype(int)

    

train.head()    

                
train['AgeBand'] = pd.cut(train['Age'], 5)

train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train.head()
train = train.drop(['AgeBand'], axis=1)

combine = [train, test]

train.head()
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['SibSp'] + dataset['Parch'] + 1 == 1, 'IsAlone'] = 1



train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train = train.drop(['Parch', 'SibSp'], axis=1)

test = test.drop(['Parch', 'SibSp'], axis=1)

combine = [train, test]



train.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_embark = train.Embarked.dropna().mode()[0]



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_embark)



train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)    
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)

    

train.head()    
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

test.head()
train['FareBand'] = pd.qcut(train['Fare'], 4)

train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

train = train.drop(['FareBand'], axis=1)

combine = [train, test]

    

ground_truth = train['Survived']

ground_truth.head()
x_train = train.drop("Survived", axis=1)

y_train = ground_truth

PassengerId = test['PassengerId']

x_test  = test.drop("PassengerId", axis=1)

x_train.shape, y_train.shape, x_test.shape
from sklearn import metrics

from sklearn import tree



clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)



clf = round(clf.score(x_train, y_train)*100,2)

clf

from sklearn.svm import SVC



svm = SVC(C=1.0, gamma='auto')

svm = svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)



svm = round(svm.score(x_train, y_train)*100,2)

svm
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)



logreg = round(logreg.score(x_train, y_train)*100,2)

logreg
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)



y_pred = random_forest.predict(x_test)



random_forest = round(random_forest.score(x_train, y_train)*100, 2)

random_forest
from sklearn.model_selection import cross_val_score



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

scores = cross_val_score(random_forest, x_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)
importances.plot.bar()

train = train.drop('IsAlone', axis=1)

test = test.drop('IsAlone', axis=1)

random_forest = RandomForestClassifier(n_estimators=100, oob_score=True)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)



random_forest = round(random_forest.score(x_train, y_train)*100,2)

print(round(random_forest,2,), "%")
param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}



from sklearn.model_selection import GridSearchCV, cross_val_score



rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

clf = GridSearchCV(estimator=rf, cv=5, param_grid=param_grid, n_jobs=-1)

clf.fit(x_train, y_train)

clf.best_params_
random_forest = RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=1, 

                                       n_jobs=-1)



random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)



random_forest.score(x_train, y_train)
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



predictions = cross_val_predict(random_forest, x_train, y_train, cv=3)

confusion_matrix(y_train, predictions)
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve



print("Precision:", precision_score(y_train, predictions))

print("Recall:",recall_score(y_train, predictions))

print("f1:", f1_score(y_train, predictions))

submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': y_pred })

submission.to_csv("submission.csv", index=False)