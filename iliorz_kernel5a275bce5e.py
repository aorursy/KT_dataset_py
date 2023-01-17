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
df = pd.read_csv('../input/train.csv')

df.describe(include='all')
df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True)
df.head(5)
df_dummy = pd.get_dummies(df)

df_dummy.head(4)
df_dummy.describe(include='all')
df_dummy['Age'].fillna(df_dummy['Age'].median(), inplace=True)

df_dummy.describe(include='all')
df_dummy.head(3)
from sklearn.model_selection import train_test_split

X = df_dummy.drop(columns=['Survived'])

y = df_dummy['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
from sklearn.ensemble import RandomForestClassifier as randomforest

rfr = randomforest(random_state=0, n_estimators=100)



rfr.fit(X_train, y_train)
rfr.score(X_test, y_test)
rfr.fit(X, y)
testdata = pd.read_csv('../input/test.csv', index_col=0)

testdata.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)

testdata_dummy = pd.get_dummies(testdata)

testdata_dummy.describe(include='all')
testdata_dummy['Age'].fillna(testdata_dummy['Age'].median(), inplace=True)

testdata_dummy['Fare'].fillna(testdata_dummy['Fare'].median(), inplace=True)
testdata_dummy.describe(include='all')
gender_submission = pd.read_csv('../input/gender_submission.csv', index_col=0)

test = pd.concat([testdata_dummy, gender_submission], axis=1)

test.head(5)
test.describe(include='all')
X_test = test.drop(columns=['Survived'])

y_test = test['Survived']
rfr.score(X_test, y_test)
y_pred = rfr.predict(X_test)

results = pd.DataFrame(X_test)

results['Survived'] = y_pred

results
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
results.to_csv('Predict.csv', columns=['Survived'])