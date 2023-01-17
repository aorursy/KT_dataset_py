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
from xgboost import XGBRegressor,XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
X = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

y = X["Survived"]

X.drop("Survived",axis=1,inplace=True)
x_train,x_valid,y_train,y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
cols_with_missing = [col for col in x_train.columns

                     if x_train[col].isnull().any()]

cols_with_missing
numeric_col = "Age"

cat_col= "Embarked"
numerical_transformer = SimpleImputer()

categorical_transformer = SimpleImputer(strategy='most_frequent')
x_train["imputed_"+numeric_col] = numerical_transformer.fit_transform(x_train[[numeric_col]])

df_test["imputed_"+numeric_col] = numerical_transformer.fit_transform(df_test[[numeric_col]])

x_valid["imputed_"+numeric_col] = numerical_transformer.transform(x_valid[[numeric_col]])

x_train["imputed_"+cat_col] = categorical_transformer.fit_transform(x_train[[cat_col]])

x_valid["imputed_"+cat_col] = categorical_transformer.transform(x_valid[[cat_col]])

df_test["imputed_"+cat_col] = categorical_transformer.transform(df_test[[cat_col]])
x_train.drop(cols_with_missing,axis=1,inplace=True)

x_valid.drop(cols_with_missing,axis=1,inplace=True)

df_test.drop(cols_with_missing,axis=1,inplace=True)
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

x_train["Sex"] = OH_encoder.fit_transform(x_train[["Sex"]])

x_valid["Sex"] = OH_encoder.fit_transform(x_valid[["Sex"]])

df_test["Sex"] = OH_encoder.fit_transform(df_test[["Sex"]])
feature_cols = ["Pclass","Sex"]
x_train = x_train[feature_cols]

x_valid = x_valid[feature_cols]

x_test = df_test[feature_cols]
titanic_model = XGBClassifier(n_estimators=700,learning_rate=0.05,n_jobs=4)
titanic_model.fit(x_train, y_train, 

             early_stopping_rounds=10, 

             eval_set=[(x_valid, y_valid)])
pred_test = titanic_model.predict(x_test,ntree_limit=titanic_model.best_ntree_limit)
output = pd.DataFrame({'PassengerId': df_test['PassengerId'],

                       'Survived': pred_test})

output.to_csv('submission.csv', index=False)