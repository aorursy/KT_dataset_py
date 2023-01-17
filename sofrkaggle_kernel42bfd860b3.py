# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train_t = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

gs=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

ob = pd.concat([train, test], ignore_index=True, sort = False)
train.fillna(train.mean(),inplace=True)
for i in range(1309):

    if ob.Sex[i]=='female':

        ob.Sex[i]=0

    elif ob.Sex[i]=='male':

        ob.Sex[i]=1 

for i in range(891):

    if train.Sex[i]=='female':

        train.Sex[i]=0

    elif train.Sex[i]=='male':

        train.Sex[i]=1 

for i in range(418):

    if test.Sex[i]=='female':

        test.Sex[i]=0

    elif test.Sex[i]=='male':

        test.Sex[i]=1
train.drop("Fare",axis=1, inplace=True)

train.drop("Ticket",axis=1, inplace=True)

train.drop("Cabin",axis=1, inplace=True)

train.drop("Embarked",axis=1, inplace=True)

train.drop("SibSp",axis=1, inplace=True)

train.drop("Parch",axis=1, inplace=True)

train.drop("Name",axis=1, inplace=True)

train.drop("Survived",axis=1, inplace=True)

train.Age.fillna(train.Age.mean(), inplace=True)

test.drop("Fare",axis=1, inplace=True)

test.drop("Ticket",axis=1, inplace=True)

test.drop("Cabin",axis=1, inplace=True)

test.drop("Embarked",axis=1, inplace=True)

test.drop("SibSp",axis=1, inplace=True)

test.drop("Parch",axis=1, inplace=True)

test.drop("Name",axis=1, inplace=True)

test.Age.fillna(train.Age.mean(), inplace=True)
clf = KNeighborsClassifier()

print(clf)

x_train, x_test, y_train, y_test = train_test_split(train, train_t.Survived, test_size=0.2)

clf.fit(x_train, y_train)

print("RF Accuracy: "+repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")

result_rf = cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')

print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))
clf_grid = GridSearchCV(clf, param_grid={'n_neighbors': range(1,50)}, cv=10)

clf_grid.fit(x_train, y_train)
clf_grid.score(x_train, y_train)
clf_grid1 = GridSearchCV(clf, param_grid={'n_neighbors': range(1,30), 'weights':['uniform','distance']}, cv=15)

clf_grid1.fit(x_train, y_train)
clf_grid1.score(x_train, y_train)
clf_grid1.best_estimator_, clf_grid1.best_params_, clf_grid1.best_score_
print(accuracy_score(y_test, clf_grid.best_estimator_.predict(x_test)))

print(accuracy_score(y_test, clf_grid1.best_estimator_.predict(x_test)))
res=clf_grid.predict(test)

print(res)
testik=test["PassengerId"]

dff=pd.DataFrame({"PassengerId": testik, "Survived": res})

display(dff)

dff.to_csv("submission.csv", index=False)