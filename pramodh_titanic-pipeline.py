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
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from xgboost import XGBRFClassifier

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

data = pd.read_csv("/kaggle/input/titanic/train.csv", index_col="PassengerId")

testData = pd.read_csv("/kaggle/input/titanic/test.csv", index_col="PassengerId")

y = data['Survived']

X = data.drop(['Survived'], axis = 1)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = testData[my_cols].copy()
numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps = [

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

preprocessor = ColumnTransformer([

    ('num', numerical_transformer, numerical_cols),

    ('cat', categorical_transformer, categorical_cols)

])
model = XGBRFClassifier()

my_pipeline = Pipeline(steps = [

    ('preprocessor', preprocessor),

    ('model', model)

])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

print(accuracy_score(y_valid, preds))

preds_Test = my_pipeline.predict(X_test)

outputOH_Pipeline = pd.DataFrame({'PassengerId' : X_test.index, 'Survived': preds_Test})

outputOH_Pipeline.to_csv('outputOH_Pipeline.csv', index=False)