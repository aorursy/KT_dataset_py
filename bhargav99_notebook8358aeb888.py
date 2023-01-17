# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train=pd.read_csv('../input/train.csv')

#print(train.head())

test=pd.read_csv('../input/test.csv')

embark_dummies_titanic  = pd.get_dummies(train['Embarked'])

#print(embark_dummies_titanic)

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



train = train.join(embark_dummies_titanic)

test    = test.join(embark_dummies_test)



train.drop(['Embarked'], axis=1,inplace=True)

test.drop(['Embarked'], axis=1,inplace=True)

test["Fare"].fillna(test["Fare"].median(), inplace=True)

X_train=train[['Pclass','Fare','C','Q']]

Y_train=train['Survived']

X_test=test[['Pclass','Fare','C','Q']]

print(X_train.info())

print(X_test.info())

logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)

print(Y_pred)

logreg.score(X_train, Y_train)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.