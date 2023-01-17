# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')
age_avg = train['Age'].mean()
fare_avg = train['Fare'].mean()

def prep(train):
    train = pd.concat([train, pd.get_dummies(train['Embarked'])], axis=1, sort=False)
    train = pd.concat([train, pd.get_dummies(train['Sex'])], axis=1, sort=False)
    
    cols_to_drop = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    train.drop(columns = cols_to_drop, inplace = True)
    
    train['Age'].fillna(age_avg, inplace = True)
    train['Fare'].fillna(fare_avg, inplace = True)
    
    return train
test.isnull().sum()
train = prep(train)
test = prep(test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

y = train['Survived']
X = train.drop(columns = ['Survived'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

for medziu_kiekis in range(100,800,100):
    for gylis in range(5,20,5):
        clf = RandomForestClassifier(n_estimators = medziu_kiekis, max_depth = gylis)
        clf.fit(X_train, y_train)
        print(medziu_kiekis, gylis, clf.score(X_test, y_test))

final_clf = RandomForestClassifier(n_estimators = 200, max_depth = 10)

final_clf.fit(X, y)

final_clf.predict(test)
gender_submission['Survived'] = final_clf.predict(test)
gender_submission.to_csv('ats1.csv', index = False)


