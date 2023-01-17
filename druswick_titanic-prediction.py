# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.drop(['PassengerId', 'Name', 'Cabin', 'Embarked', 'Ticket', 'Fare'], axis=1, inplace=True)
train['Male'] = pd.get_dummies(train['Sex'])['male']

train.drop('Sex', axis=1, inplace=True)
train['Child'] = train.apply(lambda row: row['Age'] <= 12.0, axis=1)

train.drop('Age', inplace=True, axis=1)
train['SibsSp'] = train.apply(lambda row: row['SibSp'] > 0, axis=1)

train['Parch'] = train.apply(lambda row: row['Parch'] > 0, axis=1)
labels = train['Survived']

train.drop('Survived', axis=1, inplace=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold
splits = KFold(n_splits=5, shuffle=True)

for model in [RandomForestClassifier(), SVC(), DecisionTreeClassifier(), GaussianNB()]:

    print(model)

    print(np.mean([model.fit(train.iloc[tr], labels.iloc[tr]).score(train.iloc[te], labels.iloc[te]) for tr, te in splits.split(train, labels)]))
test = pd.read_csv('../input/test.csv')

test.drop(['Name', 'Cabin', 'Embarked', 'Ticket', 'Fare'], axis=1, inplace=True)

test['Male'] = pd.get_dummies(test['Sex'])['male']

test.drop('Sex', axis=1, inplace=True)

test['Child'] = test.apply(lambda row: row['Age'] <= 12.0, axis=1)

test.drop('Age', inplace=True, axis=1)

test['SibsSp'] = test.apply(lambda row: row['SibSp'] > 0, axis=1)

test['Parch'] = test.apply(lambda row: row['Parch'] > 0, axis=1)

submission = pd.DataFrame({

        "PassengerId": test['PassengerId'],

        "Survived": SVC().fit(train, labels).predict(test.drop('PassengerId', axis=1))

    })

submission.to_csv('titanic.csv', index=False)