import pandas as pd

import numpy as np

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor



import matplotlib.pyplot as plt

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 



iowa_file_path = '../input/train.csv'



X_full = pd.read_csv(iowa_file_path)

XX = X_full.dropna(axis=0,subset=['SalePrice'])

y = X_full.SalePrice

XX.drop(["SalePrice","Id"], axis=1,inplace=True) 



X = pd.get_dummies(XX)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



my_imputer = SimpleImputer(strategy="constant")

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))

imputed_X_valid = pd.DataFrame(my_imputer.transform(val_X))



# Fill in the lines below: imputation removed column names; put them back

imputed_X_train.columns = train_X.columns

imputed_X_valid.columns = val_X.columns



rf_model = XGBRegressor(random_state=0,n_estimators=1000,n_jobs=-1,learning_rate=0.05,subsample=0.7,colsample_bytree=0.7)



tx = imputed_X_train

vx = imputed_X_valid

rf_model.fit(tx, train_y)

rf_val_predictions = rf_model.predict(vx)

print("MAE on test set: {:,.0f}".format(mean_absolute_error(rf_val_predictions, val_y)))

print("accuracy on training set {:.2f}".format(rf_model.score(tx, train_y)))

print("accuracy on validation set {:.2f}".format(rf_model.score(vx, val_y)))
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data =  XGBRegressor(random_state=0,n_estimators=1000,n_jobs=-1,learning_rate=0.05,subsample=0.7,colsample_bytree=0.7)



imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X.columns = X.columns



rf_model_on_full_data.fit(imputed_X, y)

test_data_path = '../input/test.csv'



# read test data file using pandas

td = pd.read_csv(test_data_path)



test_X = pd.get_dummies(td)

test_X = test_X.loc[:,X.columns]

assert test_X.shape[1]==X.shape[1]

test_X.fillna(0,inplace=True)



test_preds = rf_model_on_full_data.predict(test_X)



output = pd.DataFrame({'Id': td.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)


