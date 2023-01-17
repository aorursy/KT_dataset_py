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

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv("/kaggle/input/titanic/test.csv")



#Drop features we are not going to use

train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)



#Look at the first 3 rows of our training data

train.head(3)
import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',data=train,hue='Sex')
sns.boxplot(x='Pclass',y='Age',data=train)
def impute_age(cols):

    age = cols[0]

    pclass = cols[1]

    if pd.isnull(age):

        if pclass==1:

            return 38

        elif pclass==2:

            return 29

        else:

            return 24

    else:

        return age

        

    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),cmap='viridis')
sns.heatmap(test.isnull())
trainDataSex = pd.get_dummies(train['Sex'],drop_first=True)

testDataSex = pd.get_dummies(test['Sex'],drop_first=True)
train = pd.concat([train,trainDataSex],axis=1)

test = pd.concat([test,testDataSex],axis=1)

train.head()
train.drop('Sex',inplace=True,axis=1)

test.drop('Sex',inplace=True,axis=1)
train.head()
#Features

X = train.drop('Survived',axis=1)



#Label

y = train['Survived']
from sklearn.linear_model import LogisticRegression



regression = LogisticRegression()

regression.fit(X,y)
prediction = regression.predict(test)
acc_logregression = round(regression.score(X, y) * 100, 2)

acc_logregression
from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier()

rf.fit(X,y)

y_pred=rf.predict(test)

acc_rf = round(rf.score(X,y) * 100,2)

acc_rf
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })



submission.to_csv('../output/submission.csv', index=False)