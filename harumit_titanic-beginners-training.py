import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.tail()
train.info()
train.isnull().sum()
train.describe()
train.Age = train.Age.fillna(train.Age.median()) ##mean()?
train.Sex = train.Sex.replace(['male', 'female'], [0, 1])
train.Embarked = train.Embarked.fillna("S")
train.Embarked = train.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
train.info()
train.describe()
train.corr()
train.corr().style.background_gradient().format('{:.2f}')
from sklearn.model_selection import train_test_split

train = train.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
train_X = train.drop('Survived', axis=1)
train_y = train.Survived
(train_X, test_X ,train_y, test_y) = train_test_split(train_X, train_y, test_size = 0.3, random_state = 0)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)
from sklearn.metrics import (roc_curve, auc, accuracy_score)

pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
auc(fpr, tpr)
accuracy_score(pred, test_y)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
auc(fpr, tpr)
accuracy_score(pred, test_y)
from sklearn.model_selection import GridSearchCV

parameters = {
        'max_depth'         : [5, 10, 15, 20, 25],
        'random_state'      : [0],
        'n_estimators'      : [100, 150, 200, 250, 300, 400],
        'min_samples_split' : [15, 20, 25, 30, 35, 40, 50],
        'min_samples_leaf'  : [1, 2, 3],
        'bootstrap'         : [False],
        'criterion'         : ["entropy"]
}

gsc = GridSearchCV(RandomForestClassifier(), parameters,cv=3)
gsc.fit(train_X, train_y)

model = gsc.best_estimator_
gsc.best_params_
test.head()
test.tail()
test.info()
test.isnull().sum()
test.Age = test.Age.fillna(test.Age.median()) ##mean()?
test.Sex = test.Sex.replace(['male', 'female'], [0, 1])
test.Embarked = test.Embarked.fillna("S")
test.Embarked = test.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
test.Fare = test.Fare.fillna(test.Fare.median()) ##mean()?
test.isnull().sum()
test.describe()
test_data = test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
pred3 = model.predict(test_data)
pred3
submission = pd.DataFrame({
        "PassengerId": test.PassengerId,
        "Survived": pred3
    })
submission.to_csv('titanic_pre.csv', index=False)
