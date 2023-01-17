# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from xgboost import XGBRegressor

#Step 1 - Loading Data

import pandas as pd

train_file_path = "../input/train.csv"

house_data = pd.read_csv(train_file_path)

X = house_data.drop(["SalePrice","Id"],axis=1)



X.head()



## target data

Y = house_data.SalePrice

Y.head()



##train validation data split



from sklearn.model_selection import train_test_split



train_X, val_X, train_Y, val_Y = train_test_split(X, Y,random_state = 0)
## dealing with missing values on train, validation and test data

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import Imputer





test_data = pd.read_csv('../input/test.csv')

#test_X = test_data.select_dtypes(exclude=['object'])



#dropping columns with missing values

cols_with_missing = [col for col in train_X.columns 

                                 if train_X[col].isnull().sum() > 500]



reduced_X_train = train_X.drop(cols_with_missing, axis=1)

reduced_X_val  = val_X.drop(cols_with_missing, axis=1)

reduced_X_test = test_data.drop(cols_with_missing,axis=1)





one_hot_encoded_training_predictors = pd.get_dummies(reduced_X_train)

one_hot_encoded_val_predictors = pd.get_dummies(reduced_X_val)

one_hot_encoded_test_predictors = pd.get_dummies(reduced_X_test)

final_train, final_val = one_hot_encoded_training_predictors.align(one_hot_encoded_val_predictors,

                                                                    join='left', 

                                                                    axis=1)



final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)



my_imputer = Imputer()

imputed_X_train = my_imputer.fit_transform(final_train)

imputed_X_val = my_imputer.transform(final_val)

imputed_X_test = my_imputer.transform(final_test)

print(imputed_X_train.shape)

print(imputed_X_test.shape)
##fitting and validation test

my_model = XGBRegressor(n_estimators=1000,learning_rate=0.05,objective='reg:linear')

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(imputed_X_train, train_Y,verbose=False)
# make predictions

predictions = my_model.predict(imputed_X_val)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, val_Y)))
#feature importance weights

print(my_model.feature_importances_)

# Fit model using each importance as a threshold

from numpy import sort

import numpy as np

from sklearn.feature_selection import SelectFromModel

thresholds = sort(my_model.feature_importances_)

unique_thre = np.unique(thresholds)

for thresh in unique_thre:

    # select features using threshold

    selection = SelectFromModel(my_model, threshold=thresh, prefit=True)

    select_X_train = selection.transform(imputed_X_train)

    # train model

    selection_model = XGBRegressor(n_estimators=1000,learning_rate=0.05,objective='reg:linear')

    selection_model.fit(select_X_train, train_Y,verbose=False)

    # eval model

    select_X_test = selection.transform(imputed_X_val)

    y_pred = selection_model.predict(select_X_test)

    print("Thresh=%.3f, n=%d" % (thresh, select_X_train.shape[1]))

    print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, val_Y)))
 # select features using threshold

selection = SelectFromModel(my_model, threshold=0.0005, prefit=True)

select_X_train = selection.transform(imputed_X_train)

# train model

selection_model = XGBRegressor(n_estimators=1000,learning_rate=0.05,objective='reg:linear')

selection_model.fit(select_X_train, train_Y,verbose=False)

# eval model

select_X_val = selection.transform(imputed_X_val)

y_pred = selection_model.predict(select_X_val)





print(" n=%d" % ( select_X_train.shape[1]))

print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, val_Y)))
#Test Predictions

#test

select_X_test = selection.transform(imputed_X_test)



predicted_prices = selection_model.predict(select_X_test)



my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)