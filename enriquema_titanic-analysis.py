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
print(train)
list(train)
import matplotlib.pyplot as plt
plt.scatter(train['PassengerId'], train['Survived'])
plt.xlabel('PassengerId')
plt.ylabel('Survived')
plt.show()
train = train.drop(['PassengerId'], axis=1)
plt.scatter(train['Pclass'], train['Survived'])
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.show()
plt.scatter(train['Name'], train['Survived'])
plt.xlabel('Name')
plt.ylabel('Survived')
plt.show()
train = train.drop(['Name'], axis=1)
plt.scatter(train['Sex'], train['Survived'])
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.show()
plt.scatter(train['Age'], train['Survived'])
plt.xlabel('Ticket')
plt.ylabel('Survived')
plt.show()
train['Age'] = train['Age'].fillna(0).astype(np.int64)
plt.scatter(train['SibSp'], train['Survived'])
plt.xlabel('SibSp')
plt.ylabel('Survived')
plt.show()
train['HaveLessThanFiveBrothers'] = ''
train['HaveLessThanFiveBrothers'][train['SibSp'] < 5] = 0
train['HaveLessThanFiveBrothers'][train['SibSp'] >= 5] = 1
train['HaveLessThanFiveBrothers'] = train['HaveLessThanFiveBrothers'].astype(np.int64)
train = train.drop(['SibSp'], axis=1)
plt.scatter(train['Parch'], train['Survived'])
plt.xlabel('Parch')
plt.ylabel('Survived')
plt.show()
plt.scatter(train['Ticket'], train['Survived'])
plt.xlabel('Ticket')
plt.ylabel('Survived')
plt.show()
train = train.drop(['Ticket'], axis=1)
plt.scatter(train['Fare'], train['Survived'])
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.show()
train['Fare'] = train['Fare'].fillna(0).astype(np.int64)
train['Cabin'].unique()
train['Sector'] = train['Cabin'].str.replace('[0-9]+', '').fillna('Z').astype(str)
train['Sector'].unique()
plt.scatter(train['Sector'], train['Survived'])
plt.xlabel('Sector')
plt.ylabel('Survived')
plt.show()
train = train.drop(['Cabin'], axis=1)
plt.scatter(train['Embarked'], train['Survived'])
plt.xlabel('Embarked')
plt.ylabel('Survived')
plt.show()
train['isEmbarked'] = train['Embarked'].str.replace('C|Q|S', '1').fillna(0).astype(np.int64)

# Remove Embarked variable
train = train.drop(['Embarked'], axis=1)
plt.scatter(train['isEmbarked'], train['Survived'])
plt.xlabel('isEmbarked')
plt.ylabel('Survived')
plt.show()
train.dtypes
col_list_numeric_variables = list(train[['Age','Parch','Fare']])
train[col_list_numeric_variables] = (train[col_list_numeric_variables] - train[col_list_numeric_variables].min(axis=0)) / (train[col_list_numeric_variables].max(axis=0) - train[col_list_numeric_variables].min(axis=0))
train = pd.get_dummies(train, columns=['Pclass','Sex','Sector'])
print(train)
list(train)
#Charge test
test = pd.read_csv('../input/test.csv')

# Get the PassengerId for the future prediction
submission = test[['PassengerId']]
# PassengerId transformation
test = test.drop(['PassengerId'], axis=1)

# Name transformation
test = test.drop(['Name'], axis=1)

# Age transformation
test['Age'] = test['Age'].fillna(0).astype(np.int64)

# SibSp transformation
test['HaveLessThanFiveBrothers'] = ''
test['HaveLessThanFiveBrothers'][test['SibSp'] < 5] = 0
test['HaveLessThanFiveBrothers'][test['SibSp'] >= 5] = 1
train['HaveLessThanFiveBrothers'] = train['HaveLessThanFiveBrothers'].astype(np.int64)
test = test.drop(['SibSp'], axis=1)

# Ticket transformation
test = test.drop(['Ticket'], axis=1)

# Fare transformation
test['Fare'] = test['Fare'].fillna(0).astype(np.int64)

# Sector transformation
test['Sector'] = test['Cabin'].str.replace('[0-9]+', '').fillna('Z').astype(str)
test = test.drop(['Cabin'], axis=1)

# Embarked transformation
test['isEmbarked'] = test['Embarked'].str.replace('C|Q|S', '1').fillna(0).astype(np.int64)
test = test.drop(['Embarked'], axis=1)

# Normalize numerical values
col_list_numeric_variables = list(train[['Age','Parch','Fare']])
train[col_list_numeric_variables] = (train[col_list_numeric_variables] - train[col_list_numeric_variables].min(axis=0)) / (train[col_list_numeric_variables].max(axis=0) - train[col_list_numeric_variables].min(axis=0))
# Transform categorical values to numerical
test = pd.get_dummies(test, columns=['Pclass','Sex','Sector'])
print(test)
X_train = train.drop(['Survived'], axis=1)
Y_train = train['Survived']
X_test  = test.copy()
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

logisticregression = LogisticRegression()
logisticregression.fit(X_train, Y_train)
Y_pred = logisticregression.predict(X_test)
accuracy_logisticregression = round(logisticregression.score(X_train, Y_train) * 100, 2)
accuracy_logisticregression
correlation = pd.DataFrame(train.columns.delete(0))
correlation.columns = ['Feature']
correlation["Correlation"] = pd.Series(logisticregression.coef_[0])

correlation.sort_values(by='Correlation', ascending=False)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
accuracy_svc = round(svc.score(X_train, Y_train) * 100, 2)
accuracy_svc
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(X_train, Y_train)
Y_pred = randomforest.predict(X_test)
randomforest.score(X_train, Y_train)
acc_randomforest = round(randomforest.score(X_train, Y_train) * 100, 2)
acc_randomforest
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Random Forest'],
    'Score': [accuracy_logisticregression, accuracy_svc, acc_randomforest]})
models.sort_values(by='Score', ascending=False)
submission['Survived'] = Y_pred

print(submission)
#submission.to_csv('../submission/submission.csv', index=False)