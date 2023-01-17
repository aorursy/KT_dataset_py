# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

data.head()
print('shape:')

print(data.shape)

print('describe:')

print(data.describe())
# print columns count

print('columns count:', data.columns.size)



# print all columns names

print('\nAll colums names:')

print(data.columns)



# print all numeric columns names

print('\nNumeric colums names:')

numeric_cols = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]

print(numeric_cols)



# print all categorical columns names that cardinality < 10

low_cardinality_cols = [cname for cname in data.columns if data[cname].nunique() < 10 and 

                        data[cname].dtype == "object"]

print('\nCategorical columns with low cardinality:')

print(low_cardinality_cols)



# print all categorical columns names that cardinality >= 10

high_cardinality_cols = [cname for cname in data.columns if data[cname].nunique() >= 10 and 

                        data[cname].dtype == "object"]

print('\nCategorical columns with high cardinality:')

print(high_cardinality_cols)







# print a table with missing values

print("\nAll missing date:")

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# Remove columns from data

rmv_col = ['Name','Cabin','Ticket']

y = data.Survived

X = data.drop(['Survived'], axis=1)

X_test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

X = X.drop(rmv_col, axis=1)

X_test = X_test.drop(rmv_col, axis=1)



datasets = [X, X_test]



# missing value

for dataset in datasets:

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)



# new columns

for dataset in datasets:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0



# handle categoric data

X = pd.get_dummies(X)

X_test = pd.get_dummies(X_test)

X, X_test = X.align(X_test, join='left', axis=1)



# separete data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)



X_train.head()
# Define model

model = XGBClassifier(n_estimators=900, learning_rate=0.01,random_state=0)



# Create a Papeline

papeline_model = Pipeline(steps=[('model', model)])



# Preprocessing of training data, fit model 

papeline_model.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = papeline_model.predict(X_valid)



# Cross-validation

scores = -1 * cross_val_score(papeline_model, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print('Precision(Predict):', (1 - mean_absolute_error(y_valid, preds)) * 100)

print('Precision(Cross-validation):', (1 - scores.mean()) * 100)
# Preprocessing of training data, fit model 

papeline_model.fit(X, y)



# Preprocessing of validation data, get predictions

preds_test = papeline_model.predict(X_test)



# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_test.index,

                       'Survived': np.around(preds_test)})

output.to_csv('submission.csv', index=False)



# print result

output.head()