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
import pandas as pd

import numpy as np
train = pd.read_csv('..//input/titanic/train.csv')

test = pd.read_csv('..//input/titanic/test.csv')
train.head()
test.head()
train.info()
test.info()
train = train.drop(["PassengerId", "Name", "Ticket"], axis = 1)

test = test.drop(["PassengerId", "Name", "Ticket"], axis = 1)
import seaborn as sns
sns.countplot(x = "Survived", data = train)
train.columns
sns.barplot(x = 'Pclass', y = 'Survived', data = train)
sns.barplot(x = 'Sex', y = 'Survived', data = train)
sns.barplot(x = 'SibSp', y = 'Survived', data = train)
sns.barplot(x = 'Parch', y = 'Survived', data = train)
sns.barplot(x = 'Embarked', y = 'Survived', data = train)
database = [train, test]

for db in database:

        db["CabinBool"] = db["Cabin"].notnull().astype(int)

train = train.drop("Cabin", axis = 1)

test = test.drop("Cabin", axis = 1)
sns.barplot(x = 'CabinBool', y = 'Survived', data = train)
database = [train, test]

for db in database:

    db["Age"] = db["Age"].fillna(db["Age"].median())

    db["Fare"] = db["Fare"].fillna(db["Fare"].median())

    db["Embarked"] = db["Embarked"].map({"S":1, 'C': 2, 'Q':3})
train.info()
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
train.sample(10)
test.head()
train["Embarked"] = train["Embarked"].astype(int)

test["Embarked"] = test["Embarked"].astype(int)
for db in [train, test]:

    db["Age_Category"]  = pd.cut(db['Age'],[0, 5, 12, 19, 25, 35, 50, 100], labels = ['Child','Young', 'Teen', 'Student', 'Adult', 'Old', 'Senior'],right=False, duplicates='drop')

    db["Fare_Category"]  = pd.cut(db['Fare'],[0,12,30,80,513 ], labels = [1, 2, 3,4],right=False, duplicates='drop')
Sex_map = {"male": 0, "female": 1}

Age_map = {"Child": 0, "Young": 1, "Teen": 2, "Student": 3, "Adult": 4, "Old": 5, "Senior":6}

for db in [train, test]:

    db["Sex_Bool"] = db["Sex"].map(Sex_map)

    db["Age_Category"] = db["Age_Category"].map(Age_map)
test = test.drop(["Sex", "Age", "Fare"], axis = 1)

train = train.drop(["Sex", "Age", "Fare"], axis = 1)
test1 = test

train1 = train
test2 = test.drop(["SibSp", "Parch"], axis = 1)

train2 = train.drop(["SibSp", "Parch"], axis = 1)

    
train["Family"] = train["SibSp"] + train["Parch"] + 1

test["Family"] = test["SibSp"] + test["Parch"] + 1

test3 = test.drop(["SibSp", "Parch"], axis = 1)

train3 = train.drop(["SibSp", "Parch"], axis = 1)
target1.head()
predictors1 = train1.drop("Survived", axis = 1)

predictors2 = train2.drop("Survived", axis = 1)

predictors3 = train3.drop("Survived", axis = 1)

target1 = train1["Survived"]

target2 = train2["Survived"]

target3 = train3["Survived"]
from sklearn.model_selection import train_test_split

x_train1, x_test1, y_train1, y_test1  = train_test_split(predictors1, target1, test_size = 0.25, random_state = 0)

x_train2, x_test2, y_train2, y_test2  = train_test_split(predictors2, target2, test_size = 0.25, random_state = 0)

x_train3, x_test3, y_train3, y_test3  = train_test_split(predictors3, target3, test_size = 0.25, random_state = 0)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

gnb = GaussianNB()

pred1 = gnb.fit(x_train1, y_train1).predict(x_test1)

NB_accuracy1 = accuracy_score(y_test1, pred1, normalize = True)

pred2 = gnb.fit(x_train2, y_train2).predict(x_test2)

NB_accuracy2 = accuracy_score(y_test2, pred2, normalize = True)

pred3 = gnb.fit(x_train3, y_train3).predict(x_test3)

NB_accuracy3 = accuracy_score(y_test3, pred3, normalize = True)



NB_accuracy3
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

pred1 = knn.fit(x_train1, y_train1).predict(x_test1)

KNN_accuracy1 = accuracy_score(y_test1, pred1, normalize = True)

pred2 = knn.fit(x_train2, y_train2).predict(x_test2)

KNN_accuracy2 = accuracy_score(y_test2, pred2, normalize = True)

pred3 = knn.fit(x_train3, y_train3).predict(x_test3)

KNN_accuracy3 = accuracy_score(y_test3, pred3, normalize = True)

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

pred1 = rf.fit(x_train1, y_train1).predict(x_test1)

RF_accuracy1 = accuracy_score(y_test1, pred1, normalize = True)

pred2 = rf.fit(x_train2, y_train2).predict(x_test2)

RF_accuracy2 = accuracy_score(y_test2, pred2, normalize = True)

pred3 = rf.fit(x_train3, y_train3).predict(x_test3)

RF_accuracy3 = accuracy_score(y_test3, pred3, normalize = True)

from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

pred1 = dt.fit(x_train1, y_train1).predict(x_test1)

DT_accuracy1 = accuracy_score(y_test1, pred1, normalize = True)

pred2 = dt.fit(x_train2, y_train2).predict(x_test2)

DT_accuracy2 = accuracy_score(y_test2, pred2, normalize = True)

pred3 = dt.fit(x_train3, y_train3).predict(x_test3)

DT_accuracy3 = accuracy_score(y_test3, pred3, normalize = True)

from sklearn.svm import LinearSVC



lvsc = LinearSVC()

pred1 = lvsc.fit(x_train1, y_train1).predict(x_test1)

LSVM_accuracy1 = accuracy_score(y_test1, pred1, normalize = True)

pred2 = lvsc.fit(x_train2, y_train2).predict(x_test2)

LSVM_accuracy2 = accuracy_score(y_test2, pred2, normalize = True)

pred3 = lvsc.fit(x_train3, y_train3).predict(x_test3)

LSVM_accuracy3 = accuracy_score(y_test3, pred3, normalize = True)

from sklearn.svm import SVC



svc = SVC()

pred1 = svc.fit(x_train1, y_train1).predict(x_test1)

SVM_accuracy1 = accuracy_score(y_test1, pred1, normalize = True)

pred2 = svc.fit(x_train2, y_train2).predict(x_test2)

SVM_accuracy2 = accuracy_score(y_test2, pred2, normalize = True)

pred3 = svc.fit(x_train3, y_train3).predict(x_test3)

SVM_accuracy3 = accuracy_score(y_test3, pred3, normalize = True)

SVM_accuracy3
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

pred1 = lr.fit(x_train1, y_train1).predict(x_test1)

LR_accuracy1 = accuracy_score(y_test1, pred1, normalize = True)

pred2 = lr.fit(x_train2, y_train2).predict(x_test2)

LR_accuracy2 = accuracy_score(y_test2, pred2, normalize = True)

pred3 = lr.fit(x_train3, y_train3).predict(x_test3)

LR_accuracy3 = accuracy_score(y_test3, pred3, normalize = True)

models = pd.DataFrame({"Models" : ["Naive Bayes", "KNN", "Decision Tree", "Random Forest", "Logistic Regression", "SVM", "Linear SVM"],

                      "Scores1": [NB_accuracy1, KNN_accuracy1, DT_accuracy1, RF_accuracy1, LR_accuracy1, SVM_accuracy1, LSVM_accuracy1],

                       "Scores2": [NB_accuracy2, KNN_accuracy2, DT_accuracy2, RF_accuracy2, LR_accuracy2, SVM_accuracy2, LSVM_accuracy2],

                       "Scores3": [NB_accuracy3, KNN_accuracy3, DT_accuracy3, RF_accuracy3, LR_accuracy3, SVM_accuracy3, LSVM_accuracy3],

                      })

models
train3.info()
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

final_pred = rf.fit(predictors3, target3).predict(test3)
test = pd.read_csv('..//input/titanic/test.csv')

ids = test["PassengerId"]
Results = pd.DataFrame({"PassengerId": ids, "Survived": final_pred})
Results.sample(10)
Results.to_csv("submission1.csv", index  = False)