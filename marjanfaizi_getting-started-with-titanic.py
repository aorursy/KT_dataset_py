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
# Data import and exploration

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



train_data.head()
train_data.describe()
# Check how much data is missing for each column

print('### Missing values in training data (units in %) ###')

print(train_data.isnull().sum(axis=0)*100/len(train_data))

print('\n##### Missing values in test data (units in %) #####')

print(test_data.isnull().sum(axis=0)*100/len(test_data))
# Feauture engineering

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1



train_data['IsAlone'] = 1 #initialize to yes/1 is alone

train_data['IsAlone'].loc[train_data['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



test_data['IsAlone'] = 1

test_data['IsAlone'].loc[test_data['FamilySize'] > 1] = 0



train_data['Title'] = train_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

test_data['Title'] = test_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]







train_data['NameLen'] = train_data['Name'].apply(len)

test_data['NameLen'] = test_data['Name'].apply(len)



# Feature that tells whether a passenger had a cabin on the Titanic

train_data['HasCabin'] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test_data['HasCabin'] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# Seperate target from predictors

y = train_data.Survived

X = train_data.drop(['Survived'], axis=1).copy()
X.head()
# Drop column PassengerID, Ticket number, name and if more than 20% of the data is missing: Cabin

num_col = ['Pclass', 'NameLen', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin', 'FamilySize', 'IsAlone']

cat_col= ['Sex', 'Embarked', 'Title']

feature_col = num_col + cat_col
# For the remaining columns with missing data impute the values

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder



# Preprocessing for numerical data

numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),

                                        ('scaling', StandardScaler())

                                       ])



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))

#                                          ('label', OrdinalEncoder()),

#                                          ('scaling', StandardScaler())

                                         ])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, num_col),

                                               ('cat', categorical_transformer, cat_col),

                                              ])
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import GridSearchCV



parameters = {'model__n_estimators':[50, 60,100], 'model__max_depth':[4, 5, 6]}



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(random_state=0))

                             ])





search = GridSearchCV(my_pipeline, parameters, cv=5)

search.fit(X[feature_col], y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)

print(search.best_params_)



scores = cross_val_score(my_pipeline, X[feature_col], y, cv=5)



print("scores:\n", scores)

print("Average score (across experiments):")

print(scores.mean())
predictions = search.predict(test_data[feature_col])



output = pd.DataFrame( {'PassengerId': test_data.PassengerId, 'Survived': predictions} )

output.to_csv( 'my_submission.csv', index=False )