# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

 

# data processing

import pandas as pd

 

# data visualization

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')

test1 = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

res=pd.read_csv('../input/gender_submission.csv')

titanic=pd.concat([train, test], sort=False)
titanic.info()
titanic.select_dtypes(include='object').head()
titanic.isnull().sum()[titanic.isnull().sum()>0]
titanic.Fare=titanic.Fare.fillna(train.Fare.mean())

titanic.Fare=titanic.Fare.fillna(train.Fare.mean())

titanic.isnull().sum()[titanic.isnull().sum()>0]
def impute_cabin(col):

  Cabin = col[0]

  if type(Cabin) == str:

    return 1

  else:

    return 0



titanic['Cabin'] = titanic[['Cabin']].apply(impute_cabin, axis = 1)

titanic.isnull().sum()[titanic.isnull().sum()>0]
train.groupby(['Pclass','Sex']).Age.mean()
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    Sex=cols[2]

    if pd.isnull(Age):

        if Pclass == 1:

            if Sex=='male':

                return 41.28

            else:

                return 34.61

        elif Pclass == 2:

            if Sex=='male':

                return 30.74

            else:

                return 28.72

        else:

            if Sex=='male':

                return 26.50

            else:

                return 21.75

    else:

        return Age

    

titanic['Age'] = titanic[['Age','Pclass','Sex']].apply(impute_age, axis = 1)

titanic.isnull().sum()[titanic.isnull().sum()>0]
titanic.head()
dataset = titanic

 

sex = pd.get_dummies(dataset['Sex'],drop_first=True)

embark = pd.get_dummies(dataset['Embarked'],drop_first=True)

dataset.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

dataset = pd.concat([dataset,sex,embark],axis=1)

dataset.head()

dataset.isnull().sum()[dataset.isnull().sum()>0]
len_train=train.shape[0]

train=dataset[:len_train]

test=dataset[len_train:]

xtrain=train.drop("Survived",axis=1)

ytrain=train['Survived']

xtest=test.drop("Survived", axis=1)
regressor = LogisticRegression()

regressor.fit(xtrain, ytrain)

scores=cross_val_score(regressor,xtrain,ytrain,scoring='accuracy',cv=5)

np.mean(scores)
svc=make_pipeline(StandardScaler(),SVC(random_state=1))

r=[0.0001,0.001,0.1,1,10,50,100]

PSVM=[{'svc__C':r, 'svc__kernel':['linear']},

      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]

GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)

scores_svm=cross_val_score(GSSVM, xtrain.astype(float), ytrain,scoring='accuracy', cv=5)
np.mean(scores_svm)
model=regressor.fit(xtrain, ytrain)

pred=model.predict(xtest).astype(int)

model1=GSSVM.fit(xtrain, ytrain);

pred1=model1.predict(xtest).astype(int)

print(accuracy_score(res.drop("PassengerId",axis=1),pred))

print(accuracy_score(res.drop("PassengerId",axis=1),pred1))
output=pd.DataFrame({'PassengerId':xtest['PassengerId'],'Survived':pred})

output.to_csv('submission.csv', index=False)

output=pd.DataFrame({'PassengerId':xtest['PassengerId'],'Survived':pred1})

output.to_csv('submission1.csv', index=False)