# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

main_train_file_path = '../input/train.csv'

my_train_data = pd.read_csv(main_file_path)

"""

# pull data into target (y) and selectors (X)

train_y = my_data.SalePrice

selected_columns = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

train_X = my_data[selected_columns]



my_model = RandomForestRegressor(max_depth=2, random_state=0,

                                 n_estimators=100)

my_model.fit(train_X, train_y)



# Read the test data

test = pd.read_csv('../input/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test[selected_columns]

# Use the model to make predictions

predicted_prices = my_model.predict(test_X)



# Letsn see if  we have something sensible.

print(predicted_prices)



my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)

"""

print(my_train_data.describe())
print(my_train_data.columns)
my_train_price_data = my_train_data.SalePrice

print (my_train_price_data.head())
# Selecting multiple columns of interest in a frame using []

list_two_columns = ['1stFlrSF','2ndFlrSF']

# create frame of only two(variable) columns 

two_columns_data = my_train_data[list_two_columns]

two_columns_data.describe()
# Choosing the prediction target, this is the variable that we want to predict.

# Mainly converntion is (y).Here we want to predict Price of house

y = my_train_data.SalePrice
# What will be the varibales that will help us achieve this task. They are called Predictors, by convention this data is refered to as X

# Please refer to Data Description file to get the elaborative context of the below variables

house_price_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 

                        'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','OverallQual','OverallCond','MSSubClass','LowQualFinSF','GrLivArea']

X = my_train_data[house_price_predictors]
# Building the first model with the following four steps.

#1.  Define - This is where you define which type of model will be used

#2.  Fit - Capturing the patterns 

#3.  Predict

#4.  Evaluate - Verifying on the test data how well is model performing the Target predictions



from sklearn.ensemble import RandomForestRegressor



# Define model 

forest_model = RandomForestRegressor()



# Fitting the model

forest_model.fit(X,y)
# Now as the model is ready we will make some predictions of House Prices



print("Lets predict prices of some houses!")

print(X.head())

print("The predictions are :")

print(forest_model.predict(X.head()))
# Important aspect is Model Validation , checking how well is the model performing

# MAE (Mean Absolute Error) Mertic: Error = (Actual - Prediction), Absolute = Positive Value, Mean = Average of the errors



from sklearn.metrics import mean_absolute_error



predicted_house_prices = forest_model.predict(X)

mean_absolute_error(y,predicted_house_prices)
# Validation Data : Using split function to split on based of Train and Test data



from sklearn.model_selection import train_test_split



train_X,val_X,train_y,val_y = train_test_split(X,y, random_state = 0)



# Define Model 

forest_house_model = RandomForestRegressor()



# Fit the model

forest_house_model.fit(train_X,train_y)



# get the predicted prices of house, here we predict on next set other thean training set

val_predictions = forest_house_model.predict(val_X)

mean_absolute_error(val_y,val_predictions)

# This tab we will submit the results to competetion, so no need of splitting data we can train our model based on full validation train data

# We will predict results on test data



submission_model = RandomForestRegressor()

submission_model.fit(X,y)



# Read the test data 

test =  pd.read_csv('../input/test.csv')



# get the same columns basically same feature columns of the train data in test data to predict on 

test_X = test[house_price_predictors]



# Now predict on the test data

prediction_values = submission_model.predict(test_X)

print(prediction_values)
my_submission = pd.DataFrame({'Id' : test.Id , 'Saleprice' : prediction_values})

my_submission.to_csv('submission.csv', index = False)
my_train_data.dtypes.sample(10)
# XGBooster Algorithm 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer



XGB_data = my_train_data.copy()



XGB_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y_XGB = XGB_data.SalePrice

X_XGB = XGB_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X_XGB, test_X_XGB, train_y_XGB, test_y_XGB = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)



my_imputer = Imputer()

train_X_XGB = my_imputer.fit_transform(train_X_XGB)

test_X_XGB = my_imputer.transform(test_X_XGB)

from xgboost import XGBRegressor



my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_X_XGB, train_y_XGB, verbose=False)
# make predictions

predictions_XGB = my_model.predict(test_X_XGB)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions_XGB, test_y_XGB)))
# Model tuning for XGBOOST

# n_estimator 

# early_stopping_rounds

# learning_rate



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.10)

my_model.fit(train_X_XGB, train_y_XGB, early_stopping_rounds=20, 

             eval_set=[(test_X_XGB, test_y_XGB)], verbose=False)



# make predictions

predictions_XGBoost_tuning = my_model.predict(test_X_XGB)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error after tuning XGBoost : " + str(mean_absolute_error(predictions_XGBoost_tuning, test_y_XGB)))
# Model tuning for XGBOOST ON TEST VS TRAIN DATA

XGB_data = my_train_data.copy()

y_XGB = XGB_data.SalePrice

X_XGB = XGB_data[house_price_predictors]



# testing parameters

# Read the test data

test = pd.read_csv('../input/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.

test_X_XGB = test[house_price_predictors]



my_model = XGBRegressor()

my_model.fit(X_XGB, y_XGB, verbose=False)



# make predictions

predictions_XGBoost_tuning = my_model.predict(test_X_XGB)



from sklearn.metrics import mean_absolute_error

#print("Mean Absolute Error after tuning XGBoost : " + str(mean_absolute_error(predictions_XGBoost_tuning, test_y_XGB)))



# making submission to a competion 

my_submission_new = pd.DataFrame({'Id': test_X_XGB.Id, 'SalePrice': predictions_XGBoost_tuning})

# you could use any filename. We choose submission here

my_submission_new.to_csv('submission.csv', index=False)

print('Done completely')
 #partial Dependence Plots, these are basically used to check the dependence of a variable how it relates to the actual predictions and how predictions vary

# Here we will choose 3 predictors and check how the GradientBoosting model affects this



from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.ensemble import GradientBoostingRegressor

# get_some_data is defined in hidden cell above.

X = my_train_data[house_price_predictors]

y = my_train_data.SalePrice

# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models

# this was due to an implementation detail, and a future release will support all model types.

my_model = GradientBoostingRegressor()

# fit the model as usual

my_model.fit(X, y)

# Here we make the plot

my_plots = plot_partial_dependence(my_model,       

                                   features=[0, 2], # column numbers of plots we want to show

                                   X=X,            # raw predictors data.

                                   feature_names=['LotArea', 'Yearbuilt', 'TotRmsAbvGrd'], # labels on graphs

                                   grid_resolution=10) # number of values to plot on x axis
# Pipelines are the ones for making the code better organised

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer



my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
# We now fit and predict fused, imputations will be taken care by pipeline

# transformer : Data tranforamtion stage before fitting the model

# Models : the fitting stage when we create and make them learn accordigly

my_pipeline.fit(train_X, train_y)

predictions = my_pipeline.predict(test_X)
# Cross validation examines data in folds, that first 20% dataset training and then remaining 80% testing

# Second round 21-40% training, remianing 1-20% and 41-100% testing and so on , this gives the better metrics of the quality of model being built

# No need to split data as in Train-Test Split



from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer



my_pipeline_cross_validation = make_pipeline(Imputer(), RandomForestRegressor())

house_price_predictors_cross = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 

                        'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = my_train_data[house_price_predictors_cross]

y = my_train_data.SalePrice
# get the cross validation score

from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')

print(scores)
# taking average across the experiments

print('Mean Absolute Error %2f' %(-1 * scores.mean()))