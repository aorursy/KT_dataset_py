# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

for i in range(1460):

    if home_data['SaleCondition'][i]=='Normal':

        home_data['SaleCondition'][i]=0

    elif home_data['SaleCondition'][i]=='Abnorml':

        home_data['SaleCondition'][i]=1

    elif home_data['SaleCondition'][i]=='Partial':

        home_data['SaleCondition'][i]=2

    elif home_data['SaleCondition'][i]=='AdjLand':

        home_data['SaleCondition'][i]=3

    elif home_data['SaleCondition'][i]=='Alloca':

        home_data['SaleCondition'][i]=4

    elif home_data['SaleCondition'][i]=='Family':

        home_data['SaleCondition'][i]=5

for i in range(1460):

    if home_data['SaleCondition'][i]!=0 and home_data['SaleCondition'][i]!=1 and home_data['SaleCondition'][i]!=2 and home_data['SaleCondition'][i]!=3 and home_data['SaleCondition'][i]!=4 and home_data['SaleCondition'][i]!=5:

        print(str(i)+"\t"+home_data['SaleCondition'][i])
# Create target object and call it y

y = home_data.SalePrice



# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'SaleCondition']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=42)

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

rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

home_data.head()

# type(home_data['SaleCondition'][0])

home_data['SaleCondition']

# for i in range(1460):

#     if home_data['SaleCondition'][i]=='Normal':

#         home_data['SaleCondition'][i]=0

#     elif home_data['SaleCondition'][i]=='Abnorml':

#         home_data['SaleCondition'][i]=1

#     elif home_data['SaleCondition'][i]=='Partial':

#         home_data['SaleCondition'][i]=2

# # for i in range(1460):

# #     if home_data['SaleCondition'][i]!=0 and home_data['SaleCondition'][i]!=1 and home_data['SaleCondition'][i]!=2:

# #         print(home_data['SaleCondition'][i])
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=42)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

test_data

for i in range(1459):

    if test_data['SaleCondition'][i]=='Normal':

        test_data['SaleCondition'][i]=0

    elif test_data['SaleCondition'][i]=='Abnorml':

        test_data['SaleCondition'][i]=1

    elif test_data['SaleCondition'][i]=='Partial':

        test_data['SaleCondition'][i]=2

    elif test_data['SaleCondition'][i]=='AdjLand':

        test_data['SaleCondition'][i]=3

    elif test_data['SaleCondition'][i]=='Alloca':

        test_data['SaleCondition'][i]=4

    elif test_data['SaleCondition'][i]=='Family':

        test_data['SaleCondition'][i]=5
# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]

# valy = test_data.SalePrice



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

# mean_absolute_error(test_preds)
test_data.head()