import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Load the data

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test  = pd.read_csv("/kaggle/input/titanic/test.csv")

gendersub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")#for verification
#A view at the training data

train.shape
features=['Pclass', 'Sex', 'SibSp', 'Parch']
train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train = train.replace({'Sex' : { 'male' : 1, 'female' : 0}})

test = test.replace({'Sex' : { 'male' : 1, 'female' : 0}})
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
#defining variables

X=train[features]

y=train['Survived']

Z=test[features]

#correct predictions

ytrue = gendersub["Survived"]
lr = LogisticRegression()

lr.fit(X,y)

ypred1 = lr.predict(Z)

print("LR ",accuracy_score(ytrue, ypred1, normalize=True, sample_weight=None))

print(confusion_matrix(ytrue, ypred1))
dtc = DecisionTreeClassifier()

dtc.fit(X,y)

ypred2 = dtc.predict(Z)

print("DTC ",accuracy_score(ytrue, ypred2, normalize=True, sample_weight=None))

print(confusion_matrix(ytrue, ypred2))
rfc = RandomForestClassifier()

rfc.fit(X,y)

ypred3 = rfc.predict(Z)

print("RFC ",accuracy_score(ytrue, ypred3, normalize=True, sample_weight=None))

print(confusion_matrix(ytrue, ypred3))
knn = KNeighborsClassifier()

knn.fit(X,y)

ypred4 = knn.predict(Z)

print("KNN", accuracy_score(ytrue, ypred4, normalize=True, sample_weight=None))

print(confusion_matrix(ytrue, ypred4))
gbk = GradientBoostingClassifier()

gbk.fit(X, y)

ypred5 = gbk.predict(Z)



print("GBK", accuracy_score(ytrue, ypred5, normalize=True, sample_weight=None))

print(confusion_matrix(ytrue, ypred5))
ypred1
#Lets save our output file 

ids = test['PassengerId']

predictions = lr.predict(Z)



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)