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
train = pd.read_csv("/kaggle/input/titanic/train.csv", index_col=["PassengerId"])
test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col=["PassengerId"])
X_train = train.drop(columns="Survived")
Y = train["Survived"]
Y = Y.drop([62,830])
unq = X_train.Ticket.unique().tolist()
print(len(unq))
X_train.Ticket.value_counts()
len(X_train.Cabin.unique())
X_train.isnull().sum()
X = X_train.drop(columns=["Name","Cabin"])
# X.Embarked = X.drop.Embarked.dropna(axis=0)
X = X.drop([62,830])

test = test.drop(columns=["Name","Cabin"])
test.isnull().sum()
# X.info()
numerical_cols = ["Age"]
categorical_cols = ["Sex","Ticket","Embarked"]
X.isnull().sum()
# One hot encoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X[categorical_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(test[categorical_cols]))

OH_cols_train.index = X.index
OH_cols_valid.index = test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X.drop(categorical_cols, axis=1)
num_X_valid = test.drop(categorical_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# imputation
train_X_plus = OH_X_train.copy()
test_X_plus = OH_X_valid.copy()

my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(test_X_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = train_X_plus.columns
imputed_X_valid_plus.columns = test_X_plus.columns

X_train = imputed_X_train_plus
X_valid = imputed_X_valid_plus
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# scaled_df = scaler.transform(X_train)
# X_train = pd.DataFrame(scaled_df, columns=X_train.columns)
X_train
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X_train,Y)
import xgboost as xgb

learning_rate = [10,20,30,40,50,60]
for rate in learning_rate:
    model = xgb.XGBClassifier(learning_rate=0.16, n_estimators=10,
                              silent=True, objective='binary:logistic', booster='gbtree')

    model.fit(train_X,train_Y)
    preds = model.predict(test_X)
    score = mean_absolute_error(test_Y, preds)

    print('MAE: ', score)
import xgboost as xgb
model = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1)

model.fit(X_train,Y)
preds = model.predict(X_valid)
# score = mean_absolute_error(test_Y, preds)

# print('MAE: ', score)
output = pd.DataFrame({"PassengerId": test.index, "Survived": preds})
output.to_csv("XGB_approach_tuned2.csv", index=False)

# TO DO:
# cross-validation, hyperparameters optimization, check knn model, check ensemble learning models, do standardization or normalization