# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Loading the data set

train = pd.read_csv(r'../input/titanic/train.csv')
test = pd.read_csv(r'../input/titanic/test.csv')
#Data set

train
#Checking if there are any null values in data set

train.isna().sum().sum()
#Checking which columns has null values and their count

train.isna().sum()
#Dropping columns with more than fifty percent null values

train = train.drop(['Cabin'], axis = 1)
#Checking data type of each field

train.dtypes
#Dropping all the irrelevant columns to our prediction 

train = train.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
train.dtypes
#Converting object data types

train['Sex'].replace({'male':1, 'female':2}, inplace = True)
train['Embarked'].replace({'C':1, 'Q':2, 'S':3}, inplace = True)
train.dtypes
#Filling the missing values

train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].median())
train.isna().sum()
train
#Creating x and y with other features and Survived respectively

x = train.drop(['Survived'],axis = 1)
y = train['Survived']
#Splitting it into training and testing data set

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
#Training the model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
pred_train = model.predict(x_test)
#Creating Confusion Matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred_train)
#Plotting ROC Curve

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_train)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#Calculation Area Under the Curve (AUC)

print("Area Under the Curve: {0}".format(metrics.auc(fpr, tpr)))
acc = metrics.accuracy_score(y_test, pred_train)
print('Accuracy: ', acc)
#Now predicting for test data set

test
#Saving passenger id in a list 

PassengerId_list = test['PassengerId'].tolist()
PassengerId_list
#Dropping columns we dropped in training set

test = test.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis = 1)
#Checking data type of each field

test.dtypes
#Converting object data types

test['Sex'].replace({'male':1, 'female':2}, inplace = True)
test['Embarked'].replace({'C':1, 'Q':2, 'S':3}, inplace = True)
test.dtypes
#Checking if there are any null values in data set

test.isna().sum().sum()
#Checking which columns has null values and their count

test.isna().sum()
#Filling the missing values

test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].median())
test.isna().sum()
test
#Fitting test data on the model

pred_test = model.predict(test)
#Prediction

pd.DataFrame(pred_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId_list,
        "Survived": pred_test
    })

submission.to_csv('./submission.csv', index=False)
