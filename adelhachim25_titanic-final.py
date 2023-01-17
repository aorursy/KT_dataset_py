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
import numpy as np

import pandas as pd



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.head()
#see a summary of the training dataset

train.describe(include='all')
#see a sample of the dataset to get an idea of the variables

train.sample(5)
#check for any other unusable values

print(pd.isnull(train).sum())
test.describe(include="all")
#remove train feature because it has too many missing values

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
#remove ticket because it seems useless

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
print("Number of people embarking in Southampton (S):")

southampton = train[train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = train[train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = train[train["Embarked"] == "Q"].shape[0]

print(queenstown)
#replace 2 missing values of Embarked with S because it's by far the most popular

train.fillna({'Embarked':'S'})
#Age has too many missing values so I fill them with mean

train['Age'].mean()

train['Age']=train['Age'].fillna(train['Age'].mean())

test['Age'].mean()

test['Age']=train['Age'].fillna(train['Age'].mean())
test['Fare'].mean()

test['Fare']=test['Fare'].fillna(test['Age'].mean())
#drop the name feature since it contains no more useful information.

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
#One hot encoding for Sex

train['Sex']= pd.get_dummies(train['Sex'])

test['Sex']= pd.get_dummies(test['Sex'])
#One hot encoding for Sex

train['Embarked']= pd.get_dummies(train['Embarked'])

test['Embarked']= pd.get_dummies(test['Embarked'])
#check train data

train.head()
#check test data

test.head()
#splitting 22% of out data to test accuracy of different models

from sklearn.model_selection import train_test_split



features = train.drop(['Survived', 'PassengerId'], axis=1)

label = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.22, random_state = 0)

# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

print(accuracy_score(y_pred, y_test))
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

print(accuracy_score(y_pred, y_test))
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

print(accuracy_score(y_pred, y_test))
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_test)

print(accuracy_score(y_pred, y_test))

# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_test)

print(accuracy_score(y_pred, y_test))
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_test)

print(accuracy_score(y_pred, y_test))
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

print(accuracy_score(y_pred, y_test))
# KNN/k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(accuracy_score(y_pred, y_test))
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)

print(accuracy_score(y_pred, y_test))
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_test)

print(accuracy_score(y_pred, y_test))
#Random Forest had the best accuracy so I will use this model



#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = perceptron.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('PerceTitanic.csv', index=False)