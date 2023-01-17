# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

titanic = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(titanic['Age'].describe())



#Reading required columns

X=titanic[['Pclass','Age','Embarked', 'SibSp', 'Cabin','Sex']]

y=titanic[['Survived']]



Xt=test[['Pclass','Age', 'Embarked', 'SibSp', 'Cabin', 'Sex']]

Xt['Age'].fillna(Xt['Age'].mean(), inplace=True)

Xt['Embarked'].dropna(axis=0, inplace=True)

Xt['SibSp'].dropna(axis=0, inplace=True)

Xt['Cabin'].dropna(axis=0, inplace=True)

Xt['Sex'].dropna(axis=0, inplace=True)





#Having Survival values 0 and 1

result=y.Survived.unique()

print('Survival Values-->')

print(result)



#Filling nan values with some data

X['Age'].fillna(X['Age'].mean(), inplace=True)

X['Embarked'].dropna(axis=0, inplace=True)

X['SibSp'].dropna(axis=0, inplace=True)

X['Cabin'].dropna(axis=0, inplace=True)

X['Sex'].dropna(axis=0, inplace=True)





y=y.fillna(0)



#Embarked Encoding

le = preprocessing.LabelEncoder()

X.Embarked=pd.DataFrame(le.fit_transform(X.Embarked))



Xt.Embarked=pd.DataFrame(le.fit_transform(Xt.Embarked))





#Cabin Encoding

dummy_cabin=X['Cabin']

X=pd.concat([X,dummy_cabin], axis=1)

X=X.drop(['Cabin'], axis=1)



dummy_cabin_test=Xt['Cabin']

Xt=pd.concat([Xt,dummy_cabin_test], axis=1)

Xt=Xt.drop(['Cabin'], axis=1)



#Gender label encoding

X.Sex=pd.DataFrame(le.fit_transform(X.Sex))



Xt.Sex=pd.DataFrame(le.fit_transform(Xt.Sex))





#train and test split

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.175, random_state=42)



X_train.Embarked=X_train.Embarked.fillna(0)

X_test.Embarked=X_test.Embarked.fillna(0)



print(X_train.info())

print(y_train.info())



#using the logistics Algorithm

logistic=LogisticRegression(penalty='l2',C=.25)

logistic.fit(X_train, y_train)



print('Accuracy Score-->')

print(logistic.score(X_test, y_test))



survival=logistic.predict(Xt)

survivalProb=logistic.predict_proba(Xt)



print('Survived or Not-->')

print(survival)

d = {'PassengerId': test['PassengerId'], 'Survived': survival}

pd.DataFrame(d).set_index('PassengerId').to_csv('sub.csv')







# Any results you write to the current directory are saved as output.