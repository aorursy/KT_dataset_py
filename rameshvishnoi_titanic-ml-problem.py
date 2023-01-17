import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

train_file_path = "../input/train.csv"

test_file_path = "../input/test.csv"



train_data = pd.read_csv(train_file_path)

test_data = pd.read_csv(test_file_path)

# train_data = train_data.drop(["PassengerId"],axis=1)

print(train_data.columns)

print(test_data.columns)



# Any results you write to the current directory are saved as output.
missing_val_col = []

for col in train_data:

    if train_data[col].isnull().any():

        missing_val_col.append(col)



cardinal_col = []

for col in train_data:

    if train_data[col].nunique() < 10 and train_data[col].dtype == "object" :

        cardinal_col.append(col)

    

numeric_col = []

for col in train_data: 

    if train_data[col].dtype in ["int64","float64"]:

        if col != "Survived" :

            numeric_col.append(col)

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

train_y = train_data.Survived;

train_X = train_data[cardinal_col + numeric_col]

print(train_X.columns)

train_X = pd.get_dummies(train_X)

train_X = my_imputer.fit_transform(train_X)

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, random_state = 1)

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

import xgboost as xgb

from sklearn.metrics import mean_absolute_error

xgbModel = xgb.XGBRegressor()

randomForestModel = RandomForestRegressor()

gradientBoostingRegressor = GradientBoostingRegressor(n_estimators=100)



xgbModel.fit(train_X, train_y)

randomForestModel.fit(train_X, train_y)

gradientBoostingRegressor.fit(train_X, train_y)



xgbPrediction = xgbModel.predict(test_X)

randomForestPrediction = randomForestModel.predict(test_X)

gradientBoostingPrediction = gradientBoostingRegressor.predict(test_X)

error1 = mean_absolute_error(xgbPrediction, test_y)

error2 = mean_absolute_error(randomForestPrediction, test_y)

error3 = mean_absolute_error(gradientBoostingPrediction, test_y)



print("Validation MAE for XB Model: {:,f}".format(error1))

print("Validation MAE for Random Forest Model: {:,f}".format(error2))

print("Validation MAE for GradientBoosting Model: {:,f}".format(error3))

full_train_X = train_data[cardinal_col + numeric_col]

full_train_y = train_data.Survived

# full_train_X = full_train_X.drop(['Survived'],axis = 1)



full_train_X = pd.get_dummies(full_train_X)

full_train_X = my_imputer.fit_transform(full_train_X)



my_full_model = GradientBoostingRegressor(n_estimators=100)

my_full_model.fit(full_train_X,full_train_y)


full_test_X = test_data.drop("PassengerId", axis = 1)

full_test_X = test_data[numeric_col + cardinal_col]

full_test_X = pd.get_dummies(full_test_X)

full_test_X = my_imputer.transform(full_test_X)

prediction_output =  my_full_model.predict(full_test_X)

prediction_output = prediction_output.round().astype(int)

output = pd.DataFrame({"PassengerId":test_data.PassengerId, "Survived": prediction_output})

output.to_csv('submission3.csv', index=False)


