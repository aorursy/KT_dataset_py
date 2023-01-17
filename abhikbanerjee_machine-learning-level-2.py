import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



data_mel = pd.read_csv("../input/melb_data.csv")

print(data_mel.shape)
print(data_mel.isnull().sum())
from sklearn.cross_validation import train_test_split



# get the target and the predictor columns

target_col = data_mel.Price

print(target_col.shape)



melb_predictors = data_mel.drop(['Price'], axis=1)



# remove all the non numeric attributes to start with 

melb_num_predictors = melb_predictors.select_dtypes(exclude=['object'])

X_train,X_test, Y_train,  Y_test = train_test_split(melb_num_predictors, 

                                                    target_col, test_size=0.3, random_state=3)



print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)


#define the function to compute the MAE error and send back the score

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



def my_mae(X_trai, X_tes, Y_trai, Y_tes):

    model = RandomForestRegressor()

    model.fit(X_trai, Y_trai)

    predictions = model.predict(X_tes)

    return mean_absolute_error(predictions, Y_tes)



# drop the null columns and send the score

cols_missing = [col for col in X_train.columns if X_train[col].isnull().any()]



X_train_null_dropped = X_train.drop(cols_missing, axis=1)

X_test_null_dropped = X_test.drop(cols_missing, axis=1)

print(type(X_train_null_dropped))



print("Mean Absolute Error after dropping the Null columns")

print(my_mae(X_train_null_dropped, X_test_null_dropped, Y_train, Y_test))
# do the Imputer method to try imputing the columns

from sklearn.preprocessing import Imputer

import pandas as pd



my_imputer = Imputer()

X_train_imputed = my_imputer.fit_transform(X_train)

X_test_imputed = my_imputer.transform(X_test)



#check the shape before passing to the MAE

# print(type(X_train_imputed))

# print(X_test_imputed.shape)



print("Mean Absolute Error after Imputing the Null columns")

print(my_mae(X_train_imputed, X_test_imputed, Y_train, Y_test))
# 3rd way to add the imputed missing columns as new features to the existing ones

X_train_cpy = X_train.copy()

X_test_cpy = X_test.copy()



blank_cols = (col for col in X_train.columns if X_train[col].isnull().any())



for col in blank_cols:

    X_train_cpy[col + '_withMissing'] = X_train_cpy[col].isnull()

    X_test_cpy[col + '_withMissing'] = X_test_cpy[col].isnull()



print(X_train_cpy.shape)

print(X_test_cpy.shape)

    

#Imputation

my_imputer = Imputer()

X_train_cpy_imp = my_imputer.fit_transform(X_train_cpy)

X_test_cpy_imp = my_imputer.transform(X_test_cpy)



print("Mean Absolute Error after adding extra columns")

print(my_mae(X_train_cpy_imp, X_test_cpy_imp, Y_train, Y_test))