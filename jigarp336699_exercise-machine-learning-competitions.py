# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *



def remove_outliers(data):

    return data.loc[data['GrLivArea'] <= 4000.0]



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

len1 = len(home_data)

home_data = remove_outliers(home_data)

len2 = len(home_data)

print('Len of Data before and after processing (%d, %d)' % (len1, len2))
# Print all the column names of data in sorted order

colNames = home_data.columns.values

colNames_sorted = colNames

colNames_sorted.sort()

print(colNames_sorted)
## Enumeration Functions for different variables

import itertools



# Generate Enum Dict based on values

def genEnum(enumVals):

    enumVals = dict(enumerate(enumVals[::-1],1))

    enumDict = dict((v,k) for k,v in enumVals.items())

    return enumDict



# Hybrid Metric based on Quality and Condition

def qual_cond(quality, condition, enumVals=None):

    hybrid = []

    

    if enumVals is not None:

        EV = genEnum(enumVals)

        lenEV = len(EV)

        print(EV)

        print('Len EV: ', lenEV)

    

    for q, c in zip(quality, condition):

        if not isinstance(q, (int, float)):

            q = EV[q]

        if not isinstance(c, (int, float)):

            c = EV[c]

        if enumVals is not None:

            hybrid.append(q+(lenEV+c))

        else:

                hybrid.append(q+c)

    print(len(hybrid))

    return hybrid
overallEnumVals = ['Very Excellent', 'Excellent', 'Very Good', 'Good', 'Above Average', 'Average', 'Below Average', 'Fair', 'Poor', 'Very Poor']

overall_qual_cond = qual_cond(home_data['OverallQual'], home_data['OverallCond'], overallEnumVals)

bsmtEnum_vals = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']

bsmt_qual_cond = qual_cond(home_data['BsmtQual'], home_data['BsmtCond'], bsmtEnum_vals)

floorSF_qual_cond = qual_cond(home_data['1stFlrSF'], home_data['2ndFlrSF'])

#overall_qual_cond = qual_cond(home_data['OverallQual'], home_data['OverallCond'])
## DELETE THIS CELL (To run this code convert cell (markdown -> code)

# JUST RANDOM CELL TO TEST CODE BEFORE CONVERTING TO FUNCTION

overallEnumVals = ['Very Excellent', 'Excellent', 'Very Good', 'Good', 'Above Average', 'Average', 'Below Average', 'Fair', 'Poor', 'Very Poor']

bsmtEnum_vals = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']

bsmt_EnumList = dict(enumerate(bsmtEnum_vals[::-1]))

bsmt_EnumList = dict((v,k) for k,v in bsmt_EnumList.items())

print(bsmt_EnumList)
# DELETE THIS CELL (To run this code convert cell (markdown -> code)

# JUST RANDOM CELL TO TEST CODE BEFORE CONVERTING TO FUNCTION

overallEnumVals = ['Very Excellent', 'Excellent', 'Very Good', 'Good', 'Above Average', 'Average', 'Below Average', 'Fair', 'Poor', 'Very Poor']

bsmtEnum_vals = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']

bsmt_EnumList = dict(enumerate(bsmtEnum_vals[::-1]))

bsmt_EnumList = dict((v,k) for k,v in bsmt_EnumList.items())

print(bsmt_EnumList)
# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','BldgType','BsmtCond']

X = home_data[features]
# Based on data remarks

# Removing 5 outliers (i.e. properties that are > 4000 sq.ft.)

home_data = pd.read_csv(iowa_file_path)

print('Len before: ', len(home_data))

home_data1 = home_data.loc[home_data['GrLivArea'] <= 4000.0]

print('Len after: ', len(home_data1))
## Plotting Housing Data to see outliers

plt_X = home_data['SalePrice']

plt_Y = home_data['GrLivArea']

import matplotlib.pyplot as plt

plt.xlabel('Sale Price')

plt.ylabel('Liveable Area')

plt.scatter(plt_X, plt_Y)

plt.show()



print(len(plt_X))

print(len(plt_Y))



lst = ()



for i in range(0,len(plt_X)):

    lst[i] = [plt_X[i], plt_Y[i]]



print(len(lst))

import seaborn as sns

ax = sns.boxplot(x="sale price", y="liv area", data=lst)
## DO NOT RUN THIS BLOCK 

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(n_estimators=100,random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)
import numpy as np



test_sizes = np.arange(0.01,1,0.1)



train_test_data = {}

for test_size in test_sizes:

    full_data_train_X, full_data_val_X, full_data_train_y, full_data_val_y = train_test_split(X,y,test_size=test_size, random_state=1)

    train_test_data[test_size] = [full_data_train_X, full_data_val_X, full_data_train_y, full_data_val_y]



print('Done (test_size: %d) (data_size: %d)' % (len(test_sizes), len(train_test_data)))
def generate_model(X, y, estimator):

    # To improve accuracy, create a new Random Forest model which you will train on all training data

    model = RandomForestRegressor(n_estimators=estimator,random_state=1)

    

    # fit rf_model_on_full_data on all data from the training data

    model.fit(X,y)

    

    return model



def get_best_regression_model(models, X, y):

    mse = {}

    

    for k in models:

        mse[k] = compute_mse(models[k], X, y)

    

    # return best model per key

    min_key = min(mse, key=mse.get)

    print('Model Key: %r' % min_key)

    model = models[min_key]

    return (min_key, model)



def compute_mse(model, X, y):

    # All Models to evaluate per test_size denomination

    predictions = model.predict(X)

    

    # Perform MAE based on regressor's n_estimators

    mse = mean_absolute_error(predictions, y)

    

    return mse
estimators = [1, 10, 30, 50, 70, 100, 150, 300] # range(65, 75)



models = {}

est_models = {}

predictions = {}

mse = {}

for k in train_test_data:

    train_X, val_X, train_y, val_y = train_test_data[k]

    for est in estimators:

        est_models[est] = generate_model(train_X, train_y, est)

    est_key, est_model = get_best_regression_model(est_models, val_X, val_y)

    models[k] = {est_key: est_model}

    mse[k] = compute_mse(est_model, val_X, val_y)

print('Done (model_size: %d)' % len(models))



min_key = min(mse, key=mse.get)

best_rf_model = models[min_key]

for k in best_rf_model:

    rf_model_on_full_data = best_rf_model[k]



print('Estimator = %d' % k)



print('Done')
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)



print('Done')