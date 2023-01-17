import pandas as pd

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Some output from running this cell')

melbourne_file_path= '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data=pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())
print(data.describe())
print(data.columns)
print(melbourne_data.columns)
melbourne_price_data=melbourne_data.Price
print(melbourne_price_data.head())

columns_of_interest= ['Landsize','Propertycount']
two_columns_of_data=melbourne_data[columns_of_interest]
print(two_columns_of_data.describe())
iowa_data_YrSold=data.YrSold
print(iowa_data_YrSold.head())
iowa_columns_of_interest=['Id','MSSubClass']
iowa_columns=data[iowa_columns_of_interest]
print(iowa_columns.describe())
y=melbourne_data.Price
melbourne_predictors=['Rooms','Bathroom','Landsize','Lattitude','Longtitude']
X = melbourne_data[melbourne_predictors]

from sklearn.tree import DecisionTreeRegressor

#define model

melbourne_model = DecisionTreeRegressor()

#fit model
melbourne_model.fit(X,y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
print(data.columns)
iowa_prediction_column=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
iowa_data_predictions=data[iowa_prediction_column]

iowa_SalePrice_data=data.SalePrice
from sklearn.tree import DecisionTreeRegressor

#define model

iowa_model = DecisionTreeRegressor()

#fit model
iowa_model.fit(iowa_data_predictions,iowa_SalePrice_data)
print("Making predictions for the following 5 houses:")
print(iowa_data_predictions.head())
print("The predictions are")
print(iowa_model.predict(iowa_data_predictions.head()))
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y,predicted_home_prices)
from sklearn.model_selection import train_test_split
train_X, val_X , train_y, val_y = train_test_split(X,y,random_state=0)
#define model
melbourne_model = DecisionTreeRegressor()
#fit model
melbourne_model.fit(train_X,train_y)
#get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train,targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

#compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in[5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X,train_y, val_y)
    print("Max leaf nodes: %d \t \t Mean Absolute Error: %d" %(max_leaf_nodes,my_mae))
from sklearn.metrics import mean_absolute_error
predicted_home_Prices_iowa = iowa_model.predict(iowa_data_predictions)
mean_absolute_error(iowa_SalePrice_data,predicted_home_Prices_iowa)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
train_iowa_SalePrice_data, val_iowa_SalePrice_data , train_predicted_home_Prices_iowa, val_predicted_home_Prices_iowa = train_test_split(iowa_SalePrice_data,predicted_home_Prices_iowa,random_state=0)
#define model
iowa_model = DecisionTreeRegressor()
#fit model
iowa_model.fit(train_iowa_SalePrice_data,train_predicted_home_Prices_iowa)
#get predicted prices on validation data
val_prediction = iowa_model.predict(val_iowa_SalePrice_data)
print(mean_absolute_error(val_predicted_home_Prices_iowa ,val_prediction))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y,melb_preds))
#New model to submit through the kernel

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#read data

train=pd.read_csv('../input/train.csv')

#pull data into target(y) and predictors(X)

train_y= train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

#creating training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X,train_y)

#read the test data
test = pd.read_csv('../input/test.csv')
#treat the test data in the same way as training data. In this case,pull same columns.
test_X = test[predictor_cols]
#use the model to make predictions
predicted_prices=my_model.predict(test_X)
#we will look at the predicted prices to ensure we have something sensible
print(predicted_prices)

my_submission = pd.DataFrame({'Id':test.Id,'SalePrice':predicted_prices})
my_submission.to_csv('submission.csv',index=False)