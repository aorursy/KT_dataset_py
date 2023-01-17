import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/titanic/train.csv')
y = train['Survived']

train.drop(labels = ['Survived','PassengerId','Name','Ticket','Cabin','Embarked'], axis = 1, inplace = True)

train['Age'].fillna(train['Age'].mean(), inplace = True)

categorical_columns = ['Sex']

train = pd.get_dummies(train,columns = categorical_columns, dtype = int)

train.drop(labels = ['Sex_male'], axis = 1, inplace = True)



X = []

for column in train.columns:

    X.append(column)



X = train[X]
X.head()
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)

y_pred = model.predict(train)



#Function to convert the prediction to a one (survived) or zero (not survived)

def one_or_zero(abc):

    if (1 - abc) < (abc - 0):

        return 1

    else: 

        return 0

    

#Converting the predictions to 1 or 0:



list_of_predictions = []



for pred in y_pred:

    list_of_predictions.append(one_or_zero(pred))

    

y_pred = np.asarray(list_of_predictions)



#Accuracy of the same model it trained on

unique, counts = np.unique( np.asarray(y_pred == y), return_counts=True)

true_false_values = dict(zip(unique, counts))

accuracy = true_false_values[True]/len(np.asarray(y_pred == y))

accuracy
original_test = pd.read_csv('/kaggle/input/titanic/test.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.drop(labels = ['PassengerId','Name','Ticket','Cabin','Embarked'], axis = 1, inplace = True)

test['Age'].fillna(test['Age'].mean(), inplace = True)

categorical_columns = ['Sex']

test = pd.get_dummies(test,columns = categorical_columns, dtype = int)

test.drop(labels = ['Sex_male'], axis = 1, inplace = True)

test['Fare'].fillna(test['Fare'].mean(), inplace = True)



test_pred = model.predict(test)

list_of_predictions_test = []



for pred in test_pred:

    list_of_predictions_test.append(one_or_zero(pred))

    

test_pred = np.asarray(list_of_predictions_test)
submission = pd.DataFrame({

        "PassengerId": original_test["PassengerId"],

        "Survived": test_pred

    }) 



filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)