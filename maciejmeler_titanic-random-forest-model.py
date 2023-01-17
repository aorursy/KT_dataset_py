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



#addicional imports

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
#choose relevant features and create X and y

titanic_features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']

X = train_data[titanic_features]

y = train_data.Survived
#columns with missing data:

cols_with_missing = [col for col in X.columns

                     if X[col].isnull().any()]

print('Columns with missing values',cols_with_missing)

#columns with text values

s = (X.dtypes == 'object')

cols_with_text = list(s[s].index)

print('Columns with text:',cols_with_text)
X_proc = X.drop('Cabin', axis=1)



numerical_cols = ['Age']

categorical_cols = ['Sex','Embarked']



numerical_transformer = SimpleImputer(strategy='mean')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, numerical_cols),

    ('cat', categorical_transformer, categorical_cols)

])





model = RandomForestRegressor(n_estimators=100, random_state=0)

first_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])





scores = -1 * cross_val_score(first_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("MAE:", scores.mean())
model = RandomForestRegressor(n_estimators=100, random_state=0)

first_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])





scores = -1 * cross_val_score(first_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("MAE:", scores.mean())
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
first_pipeline.fit(X_proc,y)

X_test = test_data[titanic_features]

X_test = X_test.drop('Cabin', axis=1)

X_test.head()

predictions = first_pipeline.predict(X_test).astype(int)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('firxt_submission.csv', index=False)

print("Your submission was successfully saved!")