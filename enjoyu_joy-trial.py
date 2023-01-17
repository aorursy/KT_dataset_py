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
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import warnings
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")

test2=pd.read_csv("../input/titanic/test.csv")

titanic=pd.concat([train, test], sort=False)

len_train=train.shape[0]

print(len_train)
train.Fare=train.Fare.fillna(train.Fare.mean())

test.Fare=test.Fare.fillna(train.Fare.mean())
# Filling the missing values in Embarked with S

titanic['Embarked'] = titanic['Embarked'].fillna('S')
train.groupby(['Pclass','Sex']).Age.median()
def newage (cols):

    Pclass=cols[0]

    Sex=cols[1]

    Age=cols[2]

    if pd.isnull(Age):

        if Pclass== 1 and Sex=="female":

            return 35

        elif Pclass== 1 and Sex=='male':

            return 40

        elif Pclass== 2 and Sex=='female': 

            return 28

        elif Pclass== 2 and Sex=='male':

            return 30

        elif Pclass== 3 and Sex=='female':

            return 21.5

        elif Pclass== 3 and Sex=='male':

            return 25

    else:

        return Age
train.Age=train[['Pclass','Sex','Age']].apply(newage, axis=1)

test.Age=test[['Pclass','Sex','Age']].apply(newage, axis=1)
print(train.Age)
train.Cabin=train.Cabin.fillna("unknown")

test.Cabin=test.Cabin.fillna("unknown")
train['Family_Size']=train.SibSp+train.Parch+1

test['Family_Size']=test.SibSp+test.Parch+1
# dropping features

train.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)

test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
titanic=pd.concat([train, test], sort=False)



# 모든 범주형 변수 one-hot encoding! (sex, embarked)

titanic=pd.get_dummies(titanic)



titanic.dtypes.sort_values()
train=titanic[:len_train]

test=titanic[len_train:]
# Lets change type of target..

train.Survived=train.Survived.astype('int')

train.Survived.dtype
xtrain=train.drop("Survived",axis=1)

ytrain=train['Survived']

xtest=test.drop("Survived", axis=1)
RF=RandomForestClassifier(random_state=1)

scores_rf1=cross_val_score(RF,xtrain,ytrain,scoring='accuracy',cv=5)

np.mean(scores_rf1)
RF.fit(xtrain, ytrain)
svc=make_pipeline(StandardScaler(),SVC(random_state=1))

r=[0.0001,0.001,0.1,1,10,50,100]

PSVM=[{'svc__C':r, 'svc__kernel':['linear']},

      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]

GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)

scores_svm=cross_val_score(GSSVM, xtrain.astype(float), ytrain,scoring='accuracy', cv=5)

print(np.mean(scores_svm))
model=GSSVM.fit(xtrain, ytrain)
pred=model.predict(xtest)
output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})
output.to_csv('submission.csv', index=False)