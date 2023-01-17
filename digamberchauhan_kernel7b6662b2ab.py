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


# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

X = dataset.drop(['Survived','Name','Ticket'],axis=1)

y = dataset.iloc[:, 1].values

test=pd.read_csv('/kaggle/input/titanic/test.csv')

test=test.drop(['Name','Ticket'],axis=1)





pd.isna(test).any()

X.dtypes



# Taking care of missing data

X['Age']=X['Age'].fillna(X['Age'].mean())

test['Age']=test['Age'].fillna(test['Age'].mean())

test['Fare']=test['Fare'].fillna(test['Fare'].mean())







X['Embarked'] = X['Embarked'].fillna(X['Embarked'].value_counts().index[0])

pd.isna(test).any()

pd.isna(X).any()

X['Has_Cabin'] = X["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

X = X.drop(['Cabin'],axis=1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test=test.drop(['Cabin'],axis=1)







X['Sex']=pd.get_dummies(X['Sex'])

test['Sex']=pd.get_dummies(test['Sex'])





D=pd.get_dummies(X['Embarked'])

X = pd.concat([X, D], axis=1)

X = X.drop(['Embarked'],axis=1)

F=pd.get_dummies(test['Embarked'])

test = pd.concat([test, F], axis=1)

test = test.drop(['Embarked'],axis=1)





pd.isna(X).any()





# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

test = sc.transform(test)



# Fitting Random_Forest Calssifier to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

classifier.fit(X, y)



# Predicting the Test set results

y_pred = classifier.predict(test)







y_pred