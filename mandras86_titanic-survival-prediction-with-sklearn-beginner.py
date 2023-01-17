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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.head()
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df.head()
gender_submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
gender_submission_df.head()
print('Test shape:',test_df.shape)
print('Train shape:',train_df.shape)
#Checking for missing values in the training data
print('-------- Missing Values in train data-----------')
print(train_df.isnull().sum().sort_values(ascending=False))
print('-------- Missing Values in test data-----------')
print(test_df.isnull().sum().sort_values(ascending=False))
#We will not consider the Cabin information as it is very incomplete and the Name/Ticket columns wich is not relevant.
train_df.drop(['Cabin','Name', 'Ticket'], axis=1, inplace=True)
#Delete records with missing values
train_df.dropna(inplace=True)
print('Train shape:',train_df.shape)
#Encoding de Sex column
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df.head()
#Encoding de Embarked column
label_encoder = LabelEncoder()
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])
train_df.head()
#Util functions
def predict_model(df,features, y_label, func, test_size=0.2):
    X = df[features]
    y = df[y_label]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)
    model = func(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred), 
    return {'accuracy':accuracy, 'precision':precision,'recall':recall, 'model':model}
    
def logistic_fnc(X_train, y_train):
    logistic_regression = LogisticRegression(max_iter = 1000)
    logistic_regression.fit(X_train,y_train)
    return logistic_regression

def random_forest_classifier_fnc(X_train, y_train):
    randomForest = RandomForestClassifier(random_state = 2, criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 300)
    randomForest.fit(X_train, y_train)
    return randomForest

def print_model_results(model_results):
    for i in model_results:
        print(i)
        print(model_results[i])
features = train_df.columns.tolist()
y_label = 'Survived'
features.remove(y_label)
model_results = {}
model_results['logisticRegression'] = predict_model(train_df, features, y_label, logistic_fnc)
model_results['randomForest'] = predict_model(train_df, features, y_label, random_forest_classifier_fnc)
print_model_results(model_results)

#We will not consider the Cabin information as it is very incomplete and the Name/Ticket columns wich is not relevant.
test_df.drop(['Cabin','Name', 'Ticket'], axis=1, inplace=True)
#Delete records with missing values
label_encoder = LabelEncoder()
test_df['Sex'] = label_encoder.fit_transform(test_df['Sex'])
test_df['Embarked'] = label_encoder.fit_transform(test_df['Embarked'])
test_df['Age'] = test_df.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
test_df['Fare'] = test_df.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
print(test_df.shape)

test_df
test_df.drop(['Survived'], axis=1, inplace=True)
predicted_values = model_results['logisticRegression']['model'].predict(test_df)
test_df['Survived'] = predicted_values
test_df.info()
# Write test predictions for final submission
test_df[['PassengerId', 'Survived']].to_csv('rf_submission.csv', index = False)