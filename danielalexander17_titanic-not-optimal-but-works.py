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
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.info()
test.info()
train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)

test = test.drop(['Ticket'], axis=1)
train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)
train['Age'] = train['Age'].fillna(train['Age'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())
train['Embarked'] = train['Embarked'].fillna(method='ffill')

test['Fare'] = test['Fare'].fillna(test['Fare'].median())
def conv_Sex(x):

    if x=='male':

        return 0

    else:

        return 1



train['Sex'] = train['Sex'].apply(conv_Sex)

test['Sex'] = test['Sex'].apply(conv_Sex)
def conv_Embarked(x):

    if x=='C':

        return 0

    elif x=='Q':

        return 1

    else :

        return 2



train['Embarked'] = train['Embarked'].apply(conv_Embarked)

test['Embarked'] = test['Embarked'].apply(conv_Embarked)
train
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y = train['Survived']
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
Id_train = train['PassengerId']

train = train.drop(['PassengerId'], axis=1)

Id_test = test['PassengerId']

test = test.drop(['PassengerId'], axis=1)
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
train[train['Age']==13]
contoh = (train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))
train = train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

print(acc_knn)
k_range = range(1,25)

acc_test = []

acc_train = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    Y_pred = knn.predict(X_test)

    acc_knn = metrics.accuracy_score(y_test, Y_pred)

    acc_test.append(round(acc_knn*100))

    print(round(acc_knn*100))
for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    Y_pred = knn.predict(X_train)

    acc_knn = metrics.accuracy_score(y_train, Y_pred)

    acc_train.append(round(acc_knn*100))

    print(round(acc_knn*100))
print(acc_train, acc_test)
plt.plot(k_range, acc_train)

plt.plot(k_range, acc_test)

plt.show()
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc = metrics.accuracy_score(y_test, Y_pred)

acc_svc
svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_train)

acc_svc = metrics.accuracy_score(y_train, Y_pred)

acc_svc
# penggunaan paling tepat agar tidak bias adalah SVC

X_train = train [['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y_train = train['Survived']
svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(test)
submission = pd.DataFrame({

    "PassengerId": Id_test,

    "Survived": Y_pred

})
submission
submission.to_csv (r'...\output\submission.csv', index = False, header=True)