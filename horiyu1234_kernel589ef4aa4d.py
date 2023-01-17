# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

from sklearn import metrics

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.preprocessing import LabelEncoder





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Path of the file to read

iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'

X = pd.read_csv(iowa_file_path)



test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')





y = X.SalePrice

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

X.drop(['SalePrice'], axis=1, inplace=True)







# To keep things simple, we'll drop columns with missing values

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 

X.drop(cols_with_missing, axis=1, inplace=True)

#X_test.drop(cols_with_missing, axis=1, inplace=True)





# Split into validation and training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)



'''

# Picking good data for lgb

valid_fraction = 0.1

#home_data_srt = home_data.sort_values('SalePrice')

valid_rows = int(len(home_data) * valid_fraction)

train = home_data[:-valid_rows * 2]

#valid size == test size, last two sections of the data

valid = home_data[-valid_rows * 2:-valid_rows]

test = home_data[-valid_rows:]

'''



# All categorical columns

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_valid[col])

                   and set(X_train[col]) == set(test[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))





        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)



# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)

label_test = test.drop(bad_label_cols, axis=1)



# Apply label encoder 

myencoder = LabelEncoder()

for col in good_label_cols:

    label_X_train[col] = myencoder.fit_transform(label_X_train[col])

    label_X_valid[col] = myencoder.transform(label_X_valid[col])

    label_test[col] = myencoder.transform(label_test[col])



#print("MAE from Approach 2 (Label Encoding):") 

#print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))







selector = SelectKBest(f_classif, k=15)



X_new = selector.fit_transform(label_X_train,y_train)

#print(X_new)



selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                index=X_train.index, 

                                columns=label_X_train.columns)

selected_columns = selected_features.columns[selected_features.var() != 0]



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state = 1)



#print (label_X_train[selected_columns])

#print (y_train)



# fit your model

rf_model.fit(label_X_train[selected_columns],y_train)

#feature_cols



# Calculate the mean absolute error of your Random Forest model on the validation data

melb_preds = rf_model.predict(label_X_valid[selected_columns])

rf_val_mae = mean_absolute_error(y_valid, melb_preds)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))



label_test.fillna('0' ,inplace=True)



#print(label_test[selected_columns])



predictions = rf_model.predict(label_test[selected_columns])

#feature_cols



submission = pd.DataFrame({'Id':test['Id'],'SalePrice':predictions})



#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Home Predictions 1.csv'



submission.to_csv(filename,index=False)





# Any results you write to the current directory are saved as output.
