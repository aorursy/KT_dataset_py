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
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# features = ['LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd']
X = home_data[features]

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

from sklearn.linear_model import LinearRegression

def calculate_metrics(data, features):
    X_trimmed = data[features]
    y = data['SalePrice']
    train_X, test_X, train_y, test_y = train_test_split(X_trimmed, y, random_state=1)
    
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(train_X, train_y)
    y_pred = iowa_model.predict(test_X)
    
    val_mae = mean_absolute_error(test_y, y_pred)
    print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
    
    iowa_model2 = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model2.fit(train_X, train_y)
    y_pred = iowa_model2.predict(test_X)
    
    val_mae = mean_absolute_error(test_y, y_pred)
    print("Absolute Error with max_leaf_nodes=100: {:,.0f}".format(val_mae))
    
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    y_pred = rf_model.predict(test_X)
    
    val_mae = mean_absolute_error(test_y, y_pred)
    print("Absolute Error with Random Forest: {:,.0f}".format(val_mae))
    
    reg_model = LinearRegression()
    reg_model.fit(train_X, train_y)
    y_pred = reg_model.predict(test_X)
    
    val_mae = mean_absolute_error(test_y, y_pred)
    print("Absolute Error with Linear Regression: {:,.0f}".format(val_mae))

    
features = ['LotArea', 'OverallQual', 'BsmtFinSF1', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'GrLivArea']
##Feature Removed -> BedroomAbvGr, TotRmsAbvGrd
##Feature Added -> OverallQual

calculate_metrics(home_data, features)
#home_data['MSSubClass'].value_counts().sort_index().plot.line()
#home_data[home_data['LotArea'] < 20000].plot.scatter(x='LotArea', y='SalePrice')
home_data.plot.scatter(x='YrSold', y='SalePrice')
home_data.describe()
import matplotlib.pyplot as plt

corr_columns = ['BsmtFinSF1', 'TotalBsmtSF', 'LotArea', 'OverallQual', 'SalePrice']

corr = home_data[corr_columns].corr()
corr
import seaborn as sns
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(home_data[features], y)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
test_X['BsmtFinSF1'] = pd.DataFrame(my_imputer.fit_transform(test_data['BsmtFinSF1'].values.reshape(-1,1)))

#test_X.describe()

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)