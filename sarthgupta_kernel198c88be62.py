# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



from sklearn import svm



from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
test_df.head()
train_df.describe()
train_corr = train_df.corr()

train_corr
plt.figure(figsize=[16,10])

sns.heatmap(train_corr, annot=True)

plt.show()
sns.countplot(train_df.Sex, hue=train_df.Survived)

plt.show()
sns.countplot(train_df.Pclass, hue=train_df.Survived)

plt.show()
sns.countplot(train_df.Embarked, hue=train_df.Survived)

plt.show()
train_df.groupby('Survived').Sex.agg(len)
train_df.isnull().sum()
plt.figure(figsize=[12,8])

sns.heatmap(train_df.isnull(), cbar=False)

plt.show()
train_df.drop('Cabin', axis=1, inplace=True)

train_df.head()
train_df.isnull().sum()
train_df.Age.isnull().sum()
train_df.fillna(method='ffill', inplace=True)

# train_df.Age.isnull().sum()
train_df.isnull().sum()
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]

encodable = ["Pclass", "Sex", "Embarked"]
train_df.Age = (train_df.Age - train_df.Age.mean()) / (train_df.Age.max() - train_df.Age.min())    # Normalizing
X_data = train_df[features]

X_data = pd.get_dummies(X_data, columns=encodable)

print(X_data.head())



y_data = train_df['Survived']

print(y_data.head())
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=23)
param_grid_svc = {'C': [0.1, 1, 10, 100, 1000],

                  'gamma': ['scale', 'auto'],

                  'kernel': ['rbf', 'poly', 'sigmoid'],

                 'random_state': [23]}

svc_grid = GridSearchCV(svm.SVC(), param_grid_svc, refit=True)



svc_grid.fit(X_train, y_train)
print(svc_grid.best_params_)

print(svc_grid.best_estimator_)
svc_pred = svc_grid.predict(X_test)



print('Accuracy of Support Vector Classifier = ' + str(accuracy_score(y_test, svc_pred)*100) + '%')

print(classification_report(y_test, svc_pred))
param_grid_rf = {'n_estimators': [10, 20, 50, 75, 100, 200, 350, 500],

                 'max_features': ['auto', 'sqrt', 'log2'],

                'criterion': ['gini', 'entropy'],

                'max_depth': [x for x in range(5,50,5)],

                'random_state': [23]}

rf_grid = GridSearchCV(RandomForestClassifier(), param_grid_rf, refit=True)



rf_grid.fit(X_train, y_train)
print(rf_grid.best_params_)

print(rf_grid.best_estimator_)
rf_pred = rf_grid.predict(X_test)



print('Accuracy of Random Forest = ' + str(accuracy_score(y_test, rf_pred)*100) + '%')

print(classification_report(y_test, rf_pred))
test_df.drop('Cabin', axis=1, inplace=True)
len(test_df)
test_df.isnull().sum()
test_df.fillna(method='ffill', inplace=True)

test_df.isnull().sum()
test_df.Age = (test_df.Age - test_df.Age.mean()) / (test_df.Age.max() - test_df.Age.min())
test_data = test_df[features]

test_data = pd.get_dummies(test_data, columns=encodable)

print(test_data.head())
print(len(test_data))
svc_predictions = svc_grid.predict(test_data)

print(len(svc_predictions))
svc_output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': svc_predictions})

svc_output.to_csv('svc_output.csv', index=False)
rf_predictions = rf_grid.predict(test_data)



rf_output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': rf_predictions})

rf_output.to_csv('rf_output.csv', index=False)