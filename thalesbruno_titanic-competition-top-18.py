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
import pandas as pd

from sklearn.model_selection import train_test_split



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

sample_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")



# Check if there is any null value in Survived variable

assert any(train_data['Survived'].isnull()) == False



# Separate target data from features data

y = train_data.Survived

X = train_data.drop(['Survived'], axis=1)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, 

                                                                train_size=0.8, 

                                                                test_size=0.2, 

                                                                random_state=0)



categorical_columns = ['Sex', 'Embarked']

numerical_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']



# Keep selected columns only

my_cols = numerical_columns + categorical_columns

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = test_data[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_columns),

                                              ('num', numerical_transformer, numerical_columns)])



# Define model

model = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1, max_depth=10)

# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])



# Preprocessing of training data, fit model

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

acc = accuracy_score(y_valid, preds, normalize=False)

print(f"Accuracy: {acc}/{len(preds)} ({acc/(len(preds))*100:.2f}%)")
# Preprocessing of test data, get predictions

preds_test = my_pipeline.predict(X_test)
# Export predicted data to a CSV file

output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)