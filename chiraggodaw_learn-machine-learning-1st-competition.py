# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

from sklearn.impute import SimpleImputer



#filler values in case of missing places

my_imputer = SimpleImputer()

# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)

#home_data.dropna(axis = 0)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',

            'BedroomAbvGr','Street', 'TotRmsAbvGrd','Neighborhood','Condition1','Condition2',

            'HouseStyle','OverallQual','YearRemodAdd','OverallCond','ExterQual','Foundation',

            'GarageArea','MasVnrType','CentralAir','GarageType','SaleCondition','GarageType'

            ,'LotConfig','BldgType','WoodDeckSF','BsmtQual','BsmtFinSF1','GrLivArea','TotalBsmtSF',

           'Functional','GarageYrBlt','KitchenQual','Fireplaces']#,'GarageFinish']#,'MoSold']#,'MiscVal']#,'PavedDrive','GarageCars']#,'BsmtFullBath','BsmtHalfBath']#,'BsmtFinSF1'

            #,'TotalBsmtSF']#,'OpenPorchSF','FireplaceQu']#,'Exterior1st','Exterior2nd','BsmtFullBath','YrSold',

            #]

            #,'MasVnrArea','MSSubClass']



X1 = home_data[features]

X2 = pd.get_dummies(X1)

test_X1 = test_data[features]

test_X2 = pd.get_dummies(test_X1)

X, test_X = X2.align(test_X2,join ='inner', axis= 1)

my_imputer = SimpleImputer()

X = my_imputer.fit_transform(X)

test_X = my_imputer.fit_transform(test_X)







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

m = 1000000

#for i in range(2,1000):

#    iowa_model = RandomForestRegressor(n_estimators = 80,max_leaf_nodes = i, random_state = 1)

#    iowa_model.fit(train_X, train_y)

#    val_predictions = iowa_model.predict(val_X)

#    if(m> mean_absolute_error(val_predictions, val_y)):

#        max_leaf_node = i

#        print(i)

#        m = mean_absolute_error(val_predictions, val_y)

max_leaf_node = 368

print(max_leaf_node)

iowa_model = DecisionTreeRegressor(max_leaf_nodes= max_leaf_node, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(n_estimators = 80,max_leaf_nodes = max_leaf_node,random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(n_estimators = 80,

                                              max_leaf_nodes = max_leaf_node,random_state = 1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

# path to file you will use for predictions

#test_data_path = '../input/test.csv'



# read test data file using pandas

#test_data = pd.read_csv(test_data_path)





# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

#test_X1 = test_data[features]

#test_X = pd.get_dummies(test_X1)

# make predictions which we will submit. 



test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)