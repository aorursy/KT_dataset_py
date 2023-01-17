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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head(10)
train.info()
y = train['Survived']

train.drop(labels = ['Survived','PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
train.head()
train.info()
train['Age'].fillna(train['Age'].mean(), inplace = True)
train.info()
categorical_columns = ['Sex','Embarked']

train = pd.get_dummies(train,columns = categorical_columns, dtype = int)
train.info()
X = []

for column in train.columns:

    X.append(column)

X = train[X]

X
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)
y_pred = model.predict(train)

y_pred
print(y_pred.max())

print(y_pred.min())
def one_or_zero(abc):

    if (1 - abc) < (abc - 0):

        return 1

    else: 

        return 0
list_of_predictions = []



for pred in y_pred:

    list_of_predictions.append(one_or_zero(pred))

    

y_pred = np.asarray(list_of_predictions)

y_pred
unique, counts = np.unique( np.asarray(y_pred == y), return_counts=True)

true_false_values = dict(zip(unique, counts))

accuracy = true_false_values[True]/len(np.asarray(y_pred == y))

accuracy
original_test = pd.read_csv('/kaggle/input/titanic/test.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.drop(labels = ['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)

test['Age'].fillna(test['Age'].mean(), inplace = True)

categorical_columns = ['Sex','Embarked']

test = pd.get_dummies(test,columns = categorical_columns, dtype = int)
test.info()
test['Fare'].fillna(test['Fare'].mean(), inplace = True)
test.info()
test_pred = model.predict(test)

list_of_predictions_test = []



for pred in test_pred:

    list_of_predictions_test.append(one_or_zero(pred))

    

test_pred = np.asarray(list_of_predictions_test)

test_pred
submission = pd.DataFrame({

        "PassengerId": original_test["PassengerId"],

        "Survived": test_pred

    }) 



filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)