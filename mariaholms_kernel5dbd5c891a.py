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
X = pd.read_csv("/kaggle/input/titanic/train.csv")
s = X.pivot_table(index="Sex",values="Survived")

s
y = X['Survived']

X.head(10)
%matplotlib inline

from matplotlib import pyplot as plt

s.plot.bar()

plt.show()
P = X.pivot_table(index="Pclass",values="Survived")

P.plot.bar()

plt.show()
SP = X.pivot_table(index="SibSp",values="Survived")

SP.plot.bar()

plt.show()
X.drop(["Survived","Name","PassengerId","Ticket"], axis=1, inplace=True)
X.info()
X.drop(['Cabin'], axis=1, inplace=True)

X['Embarked'].describe()
X['Age'].fillna(X['Age'].median(), inplace=True)

X['Embarked'].fillna('S', inplace=True)

X.info()
X = pd.concat([X,  pd.get_dummies(X['Embarked'], prefix="Embarked")],  axis=1)

X.drop(['Embarked'], axis=1, inplace=True)

X['Sex'] = pd.factorize(X['Sex'])[0]

X.info()
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X, y);

print("Правильность на обучающем наборе:")

clf.score(X, y)
from sklearn.metrics import confusion_matrix

confusion_matrix(y,clf.predict(X))
from sklearn.metrics import roc_curve

y_pred = clf.predict_proba(X)[:, 1]

fpr, tpr, _ = roc_curve(y, y_pred)

plt.plot(fpr, tpr)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.show()
X_test = pd.read_csv("/kaggle/input/titanic/test.csv")

X_test.head(10)
PId = X_test['PassengerId']

X_test.drop(["Name","PassengerId","Ticket", "Cabin"], axis=1, inplace=True)





X_test['Sex'] = pd.factorize(X_test['Sex'])[0]

X_test['Embarked']= pd.factorize(X_test['Embarked'])[0]



X_test = pd.concat([X_test,  pd.get_dummies(X_test['Embarked'], prefix="Embarked")],  axis=1)

X_test.drop(['Embarked'], axis=1, inplace=True)



X_test['Age'].fillna(X_test['Age'].median(), inplace = True)

X_test['Fare'].fillna(X_test['Fare'].median(), inplace = True)



X_test.info()
y_test = clf.predict(X_test)

output = pd.DataFrame({"PassengerId": PId, "Survived": y_test})

output.PassengerId = output.PassengerId.astype(int)

output.Survived = output.Survived.astype(int)

output.to_csv("output.csv", index = False)

output.head(10)