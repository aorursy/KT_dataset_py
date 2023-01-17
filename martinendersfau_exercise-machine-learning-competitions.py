# Config

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

from scipy.stats import uniform, randint

import graphviz

from xgboost import XGBRegressor

import xgboost as xgb

import numpy as np





# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



# path to file you will use for predictions

test_data_path = '../input/test.csv'
# Load Train Data

home_data = pd.read_csv(iowa_file_path)



# Load Test Data for Submission

test_data = pd.read_csv(test_data_path)
home_data.head()
print("TRAIN DATA")

print("# of datasets:", len(home_data))

print("# of columns:", len(home_data.columns))
print("TEST DATA")

print("# of datasets:", len(test_data))

print("# of columns:", len(test_data.columns))
print("Comparison of missing values in TRAIN and TEST DATA")

missing_val_count_by_column = home_data.isnull().sum()

print("Number of Missing Values per Column TRAIN DATA")

print(missing_val_count_by_column[missing_val_count_by_column > 0])



missing_val_count_by_column = test_data.isnull().sum()

print("Number of Missing Values per Column TEST DATA")

print(missing_val_count_by_column[missing_val_count_by_column > 0])
SalePrice = home_data[['SalePrice']].copy()



one_hot_encoded_home_data_predictors = pd.get_dummies(home_data)

one_hot_encoded_test_data_predictors = pd.get_dummies(test_data)



# keep train and test data aligned

# Join 'inner' instead of 'left' --> align test and train data to the smallest denominator of columns

home_data_numeric_predictors, test_data_numeric_predictors = one_hot_encoded_home_data_predictors.align(one_hot_encoded_test_data_predictors,join='inner',axis=1)
print("Train Data:", len(home_data_numeric_predictors.columns))

print("Test Data:", len(test_data_numeric_predictors.columns))
print("Comparison of missing values in TRAIN and TEST DATA")

missing_val_count_by_column = home_data_numeric_predictors.isnull().sum()

print("Number of Missing Values per Column TRAIN DATA")

print(missing_val_count_by_column[missing_val_count_by_column > 0])



missing_val_count_by_column = test_data_numeric_predictors.isnull().sum()

print("Number of Missing Values per Column TEST DATA")

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# impute missing values



def my_impute_missing_values(home_data_numeric_predictors):

    new_home_data = home_data_numeric_predictors.copy()



    cols_with_missing = [col for col in new_home_data.columns if new_home_data[col].isnull().any()]

    for col in cols_with_missing:

        new_home_data[col + '_was_missing'] = new_home_data[col].isnull()

    

    my_imputer = SimpleImputer()

    new_data = pd.DataFrame(my_imputer.fit_transform(new_home_data))

    

    new_data.columns = new_home_data.columns

    return new_data



new_home_data = my_impute_missing_values(home_data_numeric_predictors)

new_test_data = my_impute_missing_values(test_data_numeric_predictors)

#Align train (home) and test data

new_home_data, new_test_data = new_home_data.align(new_test_data,join='inner',axis=1)
new_home_data['LotArea'] = np.log(new_home_data['LotArea'])

new_test_data['LotArea'] = np.log(new_test_data['LotArea'])
new_home_data['GrLivArea'] = np.log(new_home_data['GrLivArea'])

new_test_data['GrLivArea'] = np.log(new_test_data['GrLivArea'])
for x in ['TotalBsmtSF','OverallQual']:

    new_home_data[x] = np.log(new_home_data[x])

    new_test_data[x] = np.log(new_test_data[x])
list(new_home_data.columns)
# Create target object and call it y



y = SalePrice



# Create X

features = [x for x in list(new_home_data.columns) if x != 'Id']

features = list(set(features))

X = new_home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y)

len(features)
# Current best submission

parameter = {'learning_rate': 0.03, 'random_state': 2, 'n_estimators': 2000}

xgb_model = XGBRegressor(**parameter)

xgb_model.fit(train_X, train_y, early_stopping_rounds=30, eval_set=[(val_X, val_y)], verbose=False)

xgb_val_predictions = xgb_model.predict(val_X)

xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)



print("Validation MAE for XGB Model: {:,.0f}".format(xgb_val_mae))

xgb_model
def report_best_scores(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
# Parameter Tuning



params = {

    "colsample_bytree": uniform(0.7, 0.3),

    "gamma": uniform(0, 0.5),

    "learning_rate": uniform(0.03, 0.3), # default 0.1 

    "max_depth": randint(2, 6), # default 3

    "n_estimators": randint(100, 1000), # default 100

    "subsample": uniform(0.6, 0.4),

}



#Comment in to find parameters (runs ca. 10 minutes)

#xgb_model = XGBRegressor()

#search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=10, return_train_score=True)

#search.fit(train_X,train_y)

#report_best_scores(search.cv_results_, 1)
def my_plot_importance(booster, figsize, **kwargs): 

    from matplotlib import pyplot as plt

    from xgboost import plot_importance

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax, **kwargs)



my_plot_importance(xgb_model,figsize=(22,28))

xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)
# Define the model. Set random_state to 1

# First try with random forest regressor



#rf_model = RandomForestRegressor(n_estimators=100,random_state=1)

#rf_model.fit(train_X, train_y)

#rf_val_predictions = rf_model.predict(val_X)

#rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



#print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data

model_on_full_data = XGBRegressor(**parameter)



# fit rf_model_on_full_data on all data from the training data

model_on_full_data.fit(X,y)

# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = new_test_data[features]



# make predictions which we will submit. 

test_preds = model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)