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
# Load libraries

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, GridSearchCV
# get titanic & test csv files as a DataFrame

X_orig = pd.read_csv("/kaggle/input/titanic/train.csv")

X_test_orig = pd.read_csv("/kaggle/input/titanic/test.csv")



# make a copy

X = X_orig.copy()

X_test = X_test_orig.copy()



# separate target from predictors

y = X.Survived

X.drop(['Survived'], axis=1, inplace=True)
X.columns
# remove columns

X.drop(['PassengerId', 'Name',

       'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

X_test.drop(['PassengerId', 'Name',

       'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
X.head()
# Numerical and categorical columns transformers

numeric_features = ['Age', 'Fare']

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])



categorical_features = ['Sex']

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])



# Append classifier to preprocessing pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', RandomForestClassifier())])



# Create space of candidate learning algorithms and their hyperparameters

search_space = [{'classifier': [LogisticRegression(random_state = 0)],

                 'classifier__penalty': ['l1', 'l2'],

                 'classifier__C': np.logspace(0, 4, 10)},

                {'classifier': [RandomForestClassifier(random_state = 0)],

                 'classifier__n_estimators': range(10, 1000, 50),

                 'classifier__max_depth': range(1,10,1),

                }]
# Create grid search 

grid = GridSearchCV(clf, search_space, cv=5, verbose=0)
# Fit grid search

best_model = grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)

results
# View best model

best_model.best_estimator_.get_params()['classifier']
# Predict target vector

preds_test = best_model.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_test_orig.PassengerId,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)