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
import pandas as pd
train_file_path = '../input/train.csv'
train_data = pd.read_csv(train_file_path)
print(train_data.describe())
# Too many columns are there, so first we need to filter out which of them are necessary, thus use following command to check out the columns
print(train_data.columns)
# seperate out price data of homes  by using DOT notation, use the above list to search the column name
train_price_data = train_data.SalePrice
#  Head command prints the first few rows 
print(train_price_data.head())
# Selecting multiple columns of interest in a frame using []
list_two_columns = ['1stFlrSF','2ndFlrSF']
# create frame of only two(variable) columns 
two_columns_data = train_data[list_two_columns]
two_columns_data.describe()
# Chosing the prediction target, this is the variable that we want to predict. Mainly converntion is (y).Here we want to predict Price of house
y = train_data.SalePrice
# What will be the varibales that will help us achieve this task. They are called Predictors, by convention this data is refered to as X
# Please refer to Data Description file to get the elaborative context of the below variables
house_price_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                        'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','OverallQual','OverallCond','MSSubClass','LowQualFinSF','GrLivArea']
X = train_data[house_price_predictors]
# Building the first model with the following four steps.
#1.  Define - This is where you define which type of model will be used. Decision tree, RF etc
#2.  Fit - Capturing the patterns 
#3.  Predict
#4.  Evaluate - Verifying on the test data how well is model performing the Target predictions

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Define model 
iowa_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()

# Fitting the model
iowa_model.fit(X,y)


# Now as the model is ready we will make some predictions of House Prices
# We shall select Head for top 5 houses
print("Lets predict prices of some houses!")
print(X.head())
print("The predictions are :")
print(iowa_model.predict(X.head()))
# Important aspect is Model Validation , checking how well is the model performing
# MAE (Mean Absolute Error) Mertic: Error = (Actual - Prediction), Absolute = Positive Value, Mean = Average of the errors

from sklearn.metrics import mean_absolute_error

predicted_house_prices = iowa_model.predict(X)
mean_absolute_error(y,predicted_house_prices)
# Validation Data : Using split function to split on based of Train and Test data

from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y = train_test_split(X,y, random_state = 0)

# Define Model 
iowa_house_model = DecisionTreeRegressor()

# Fit the model
iowa_house_model.fit(train_X,train_y)

# get the predicted prices of house, here we predict on next set other thean training set
val_predictions = iowa_house_model.predict(val_X)
mean_absolute_error(val_y,val_predictions)
# Concepts of Overfitting, Underfitting and Model optimization
# Overfitting : In case Tree based model, when the depth of tree is too much high, the leafs would be high and thus model matches trainign data accurately.
# This catches patterns which wont be of specific use in general. Thus performs bad on testing data
# UnderFitting : Tree depth is shallow and we spilt data in very few patterns example 2-4 leaves. Thus we would fail in capturing important patterns.
# Balance has to be in capturing optimal middle ware between which performs well.

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    pred_vals = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val,pred_vals)
    return mae
    
# Now lets try different max_leaf values and identify the optimal leaf value at which mea is lowest
# Low MEA is directly proportional to model perfoming better

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
# Here we observe at max_leaf_nodes of 50, model performs better.

# Lets dive into some other model to look into the performance
# Decision tree leaves you with a complex decsion of max_leaves to perform optimally
# Too many leaves tend to cause overfitting and thus perform bad on testing data
# Few leaves cause underfitting and thus loose out crucial patterns
# RandomForest is the model which uses may trees,  and it makes a prediction by averaging the predictions of each component tree. This may be better than a single deciosn tree
# Lets check out the results

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model  = RandomForestRegressor()
forest_model.fit(X,y)
#iowa_forest_preds = forest_model.predict(val_X)
#mean_absolute_error(val_y,iowa_forest_preds)

# Thus in the result we see, default RandomForest performs better than the optimal Decision tr

# This tab we will submit the results to competetion, so no need of splitting data we can train our model based on full validation train data
# We will predict results on test data

from sklearn.ensemble import RandomForestRegressor

submission_model = RandomForestRegressor()
submission_model.fit(X,y)

# Read the test data 
test =  pd.read_csv('../input/test.csv')

# get the same columns basically same feature columns of the train data in test data to predict on 
test_X = test[house_price_predictors]

# Now predict on the test data
prediction_values = submission_model.predict(test_X)
print(prediction_values)
# Now we will prepare the submission file for the competition
my_submission = pd.DataFrame({'Id' : test.Id , 'Saleprice' : prediction_values})
my_submission.to_csv('submission.csv', index = False)
# Using Categorical values with One Hot Encoding 
train_data.dtypes.sample(10)
# XGBooster Algorithm 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

XGB_data = train_data.copy()

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
XGB_data = train_data.copy()
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
my_submission_new = pd.DataFrame({'Id': test_XGB.Id, 'SalePrice': predictions_XGBoost_tuning})
# you could use any filename. We choose submission here
my_submission_new.to_csv('submission.csv', index=False)
print('Done completely')
# partial Dependence Plots, these are basically used to check the dependence of a variable how it relates to the actual predictions and how predictions vary
# Here we will choose 3 predictors and check how the GradientBoosting model affects this

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
# get_some_data is defined in hidden cell above.
X = train_data[house_price_predictors]
y = train_data.SalePrice
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
# These graphs in the above plots give us a sense of understanding that our model is working somewhat sensible
# Trend in first graph is - As the LotArea increases the Price of the House increases
# Second, plot suggests TotalRoomAboveGrade is direcltly proportional to rate of the house, more the above grade rooms more is the price of overall house
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
X = train_data[house_price_predictors_cross]
y = train_data.SalePrice
# get the cross validation score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)
# taking average across the experiments
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
