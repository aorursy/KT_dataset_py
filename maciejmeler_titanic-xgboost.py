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

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
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



#split into training and validation data

X_train, X_valid, y_train, y_valid = train_test_split(X_proc, y, random_state = 0)



missing_cols = ['Age','Embarked']

categorical_cols = ['Sex','Embarked']



imp_X_train = X_train.copy()

imp_X_valid = X_valid.copy()



mf_imputer = SimpleImputer(strategy='most_frequent')



imp_X_train[missing_cols] = mf_imputer.fit_transform(X_train[missing_cols])

imp_X_valid[missing_cols] = mf_imputer.transform(X_valid[missing_cols])







OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imp_X_train[categorical_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(imp_X_valid[categorical_cols]))



OH_cols_train.index = imp_X_train.index

OH_cols_valid.index = imp_X_valid.index



num_X_train = imp_X_train.drop(categorical_cols, axis=1)

num_X_valid = imp_X_valid.drop(categorical_cols, axis=1)



final_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

final_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
xgbr_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgbr_model.fit(final_X_train, y_train, 

             early_stopping_rounds=10, 

             eval_set=[(final_X_valid, y_valid)],

             verbose=False)



predictions = xgbr_model.predict(final_X_valid)

mae_score = mean_absolute_error(predictions, y_valid)

print(mae_score)
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
X_test = test_data[titanic_features]

X_test = X_test.drop('Cabin', axis=1)



#preprocessing

imp_X_test = X_test.copy()

imp_X_test[missing_cols] = mf_imputer.fit_transform(X_test[missing_cols])

OH_cols_test = pd.DataFrame(OH_encoder.fit_transform(imp_X_test[categorical_cols]))

OH_cols_test.index = imp_X_test.index

num_X_test = imp_X_test.drop(categorical_cols, axis=1)

final_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
#final predictions and save

test_predictions = xgbr_model.predict(final_X_test).astype(int)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})

output.to_csv('xgboostv1.csv', index=False)

print("Your submission was successfully saved!")