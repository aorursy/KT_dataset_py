##### Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor



# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
#features = lists(home_data)
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X.as_matrix(), y.as_matrix(), random_state=1, test_size = 0.25)
"""
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
#Handling missing values-
#drop missing value Columns-
cols_with_missing = [col for col in train_X.columns
                                 if train_X[col].isnull().any()]
reduced_train_X = train_X.drop(cols_with_missing, axis=1)
reduced_val_X  = val_X.drop(cols_with_missing, axis=1)
reduced_rf_model = RandomForestRegressor(random_state=1)
reduced_rf_model.fit(reduced_train_X, train_y)
reduced_rf_val_predictions = reduced_rf_model.predict(reduced_val_X)
reduced_rf_val_mae = mean_absolute_error(reduced_rf_val_predictions, val_y)
print("Validation MAE for reduced column (dropping columns with missing values): {:,.0f}".format(reduced_rf_val_mae))
"""
#use Imputation

from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
xg_model = XGBRegressor(n_estimators = 100, learning_rate = 0.5)

myImputer = SimpleImputer()
imputed_train_X = myImputer.fit_transform(train_X)
imputed_val_X = myImputer.transform(val_X)

xg_model.fit(imputed_train_X, train_y, early_stopping_rounds = 5, eval_set=[(val_X,val_y)], verbose = False)
imputed_xg_val_predictions = xg_model.predict(imputed_val_X)
imputed_xg_val_mae = mean_absolute_error(imputed_xg_val_predictions, val_y)
print("Validation MAE for reduced column (using imputation): {:,.0f}".format(imputed_xg_val_mae))
#*********************
#one_hot_encoded_training_predictors = pd.get_dummies(imputed_train_X)
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the 

rf_model_on_full_data.fit(X,y)
#xg_model
from xgboost import XGBRegressor
xg_model_full = XGBRegressor(n_estimators = 100)
xg_model_full.fit(X,y, early_stopping_rounds = 5)
# path to file you will use for predictions
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)

test_X = test_data[features]
# make predictions which we will submit. 
xg_model.fit(X,y)
test_preds = xg_model.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)