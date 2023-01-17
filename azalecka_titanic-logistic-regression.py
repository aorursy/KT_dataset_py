import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# loading data
data = pd.read_csv('../input/train.csv')
data.head()
# how many observations and columns
data.shape
print('Data shape: ', data.shape)
print('Data type: ', data.dtypes)
data.describe()
data.isnull().sum()
data.isnull().mean()
# Sex, creating a new column male
data['Male'] = ((data['Sex'] == 'male') + 0)
# Age (null values replaced with mean)
age_mean = data['Age'].mean()
data['Age'].replace(np.nan, age_mean, inplace=True)
# Embarked (B28 has null, but B20 and B22 are in S, so we assume also B28 is S)
data['Embarked'].replace(np.nan, 'S', inplace=True)
data_embarked = pd.get_dummies(data['Embarked'])
data_embarked.columns = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
data = data.join(data_embarked)
del_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
for i in del_columns:
    del data[i]
y = data['Survived']
X = data.copy()
del X['Survived']
# scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=123,
                                                    stratify=y)
# logistic regression
penalty    = ['l1','l2']
C_range    = 2. ** np.arange(-10, 0, step=1)
parameters = [{'C': C_range, 'penalty': penalty}]

grid = GridSearchCV(LogisticRegression(), parameters, cv=5)
grid.fit(X_train, y_train)

bestC = grid.best_params_['C']
bestP = grid.best_params_['penalty']
print ("The best parameters are: cost=", bestC , " and penalty=", bestP, "\n")

print("Accuracy: {0:.3f}".format(accuracy_score(grid.predict(X_test), y_test)))
# loading original test file
original_test = pd.read_csv('../input/test.csv')
PassengerId = original_test['PassengerId']
original_test.shape
original_test.columns
# preparing test dataset to be predicted
age_mean = original_test['Age'].mean()
original_test['Age'].replace(np.nan, age_mean, inplace=True)
fare_mean = original_test['Fare'].mean()
original_test['Fare'].replace(np.nan, fare_mean, inplace=True)
original_test['Male'] = ((original_test['Sex'] == 'male') + 0)
original_data_embarked = pd.get_dummies(original_test['Embarked'])
original_data_embarked.columns = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
original_test = original_test.join(original_data_embarked)
original_del_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
for i in original_del_columns:
    del original_test[i]
original_test.isnull().sum()
scaler = StandardScaler()
scaler.fit(original_test)
original_test = scaler.transform(original_test)
survived = grid.predict(original_test)
final = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': survived })
final.to_csv('final.csv', index=False)