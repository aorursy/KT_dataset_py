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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# Reading the train data and test data
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
# Values for learning
X = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
X_to_pred = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# Fill NaN values with median values in column 'Age' and with S (will be Southampton) in column 'Embarked'
X = X.fillna({'Age' : X.Age.median(),
              'Embarked' : 'S'})
X_to_pred = X_to_pred.fillna({'Age' : X_to_pred.Age.median(),
                              'Embarked' : 'S',
                              'Fare' : X_to_pred.Fare.median()})
# Convertation string variable 'Sex' into nominative variable (femail=0. male=1)
X = X.replace(to_replace=['female','male'],value=[0, 1])
X_to_pred = X_to_pred.replace(to_replace=['female','male'],value=[0, 1])
# Convertation string variable 'Embarked' into nominative variable by get_dummies
X = pd.get_dummies(X)
X_to_pred = pd.get_dummies(X_to_pred)
# Target variable
y = train_data.Survived
# Spliting train data into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Choosing a model
clf_rf = RandomForestClassifier()
# Variation of paramers for grid_search
parametrs = {'n_estimators' : [400, 500],          
             'criterion' : ['gini', 'entropy'],
             'min_samples_split' : [4, 5, 10, 15],
             'min_samples_leaf' : [5, 10]}      
# Choosing grid_search for searching best model's parametrs and cross-validation
grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=10)
# Fiting grid_search by train data
grid_search_cv_clf.fit(X_train, y_train)
grid_search_cv_clf.best_params_
# Choosing best best parametrs
best_clf_rf = grid_search_cv_clf.best_estimator_
# Accurance score train data
best_clf_rf.score(X_train, y_train)
# Accurance score test data
best_clf_rf.score(X_test, y_test)
# Let's see which features are most important
feature_importances = best_clf_rf.feature_importances_
# Put features and their importances into DataFrame
feature_importances_df = pd.DataFrame({'feature' : list(X_train),
                                       'feature_importances' : feature_importances})
# Showing importance of features
feature_importances_df.sort_values('feature_importances', ascending=False)
# Making a binary prediction: 1 for survived, 0 for deceased)
y_pred = best_clf_rf.predict(X_to_pred)
# Making output (saving results in the csv-file)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
