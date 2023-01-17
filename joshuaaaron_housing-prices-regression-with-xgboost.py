# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt





# Input Directory:

# import os

# print(os.listdir("../input"))
test_data = pd.read_csv("../input/test.csv")

train_data = pd.read_csv("../input/train.csv")
# MSSubClass is a categorical variable, so we convert to str type

def strMS(data):

    data['MSSubClass'] = data['MSSubClass'].astype(str)

    

strMS(test_data)

strMS(train_data)
# Filling LotFrontage NaNs with the median lot size of their respective neighborhood. 



gb_neigh_LF = train_data['LotFrontage'].groupby(train_data['Neighborhood'])



# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 

# associated with it...

for key,group in gb_neigh_LF:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train_data['LotFrontage'].isnull() & (train_data['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_data.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()

    

    

# Doing the same for the test set

for key,group in gb_neigh_LF:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train_data['LotFrontage'].isnull() & (train_data['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    test_data.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()

    

    
# Dropping Columns where 97% of the data is a single class



from statistics import mode

low_var_cat = [col for col in train_data.select_dtypes(exclude=['number']) if 1 - sum(train_data[col] == mode(train_data[col]))/len(train_data) < 0.03]



train_data = train_data.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis = 1)

test_data = test_data.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis = 1)
# Converting the 5 level categoric values into numeric values



has_rank = [col for col in train_data if 'TA' in list(train_data[col])]

dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}



for col in has_rank:

    train_data[col] = train_data[col].map(dic_num)

    

for col in has_rank:

    test_data[col] = test_data[col].map(dic_num)
# TRAINING DATA:

# Create a copy, turn categorical data into dummy variables

train_data_dummied = train_data.copy()

train_data_dummied = pd.get_dummies(train_data_dummied)

# Remove the original categorical variables

keep_cols = train_data_dummied.select_dtypes(include=['number']).columns

train_data_dummied = train_data_dummied[keep_cols]

# Fill NaNs with medians

train_data_dummied = train_data_dummied.fillna(train_data_dummied.median())



# TESTING DATA:

test_data_dummied = test_data.copy()

test_data_dummied = pd.get_dummies(test_data_dummied)

# Fill NaNs with medians

test_data_dummied = test_data_dummied.fillna(test_data_dummied.median())

# Some dummy variables exist in train but not test; 

# create them in the test set and set to zero.

for col in keep_cols:

    if col not in test_data_dummied:

        test_data_dummied[col] = 0

# Remove the original categorical variables

test_data_dummied = test_data_dummied[keep_cols]
# Normalizing Input data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))



# train_X = scaler.fit_transform( train_X )

# test_X = scaler.transform( test_X )



train_data = scaler.fit_transform(train_data_dummied)

train_data = pd.DataFrame(train_data, columns=train_data_dummied.columns.values)



test_data = scaler.transform(test_data_dummied)

test_data = pd.DataFrame(test_data, columns=test_data_dummied.columns.values)



# **VARIABLES FOR REDISTRIBUTING OUTPUT**

# predictions = predictions + add_by

# predictions = predictions / div_by

add_by = abs(scaler.min_[45])

div_by = scaler.scale_[45]



# scaled_work_orders_df.to_csv('preprocessed_data.csv', index=False)



print("Data is ready for model input")

print("Output scalar:")

print(("Add by %s") % add_by)

print(("Divide by %s") % div_by)
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import cross_val_score



xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_data.drop(['SalePrice','Id'], axis = 1), train_data['SalePrice'], cv = 5, n_jobs = -1)



print('CV Score is: '+ str(np.mean(cv_score)))
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
Y_vals = train_data['SalePrice']

X_vals = train_data.drop(['SalePrice','Id'], axis = 1)



xgb_test.fit(X_vals, Y_vals)
final = test_data.drop(['SalePrice','Id'], axis = 1)

test_preds = xgb_test.predict(final)
# Redistribute output

test_preds = test_preds + add_by

test_preds = test_preds / div_by
output = pd.DataFrame({'Id': test_data_dummied.Id,

                       'SalePrice': test_preds})



output.to_csv('submission.csv', index=False)