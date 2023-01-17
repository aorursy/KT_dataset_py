# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor




# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
train_X_impute = train_X.copy()
train_y_impute = train_y.copy()
val_X_impute = val_X.copy()
val_y_impute = val_y.copy()

cols_with_missing = (cols for cols in train_X.columns if train_X[cols].isnull().any())
for cols in cols_with_missing:
    train_X_impute[cols + '_was_missing'] = train_X_impute[cols].isnull()
    val_X_impute[cols + '_was_missing'] = val_X_impute[cols].isnull()
    train_y_impute[cols + '_was_missing'] = train_y_impute[cols].isnull()

myimputer = SimpleImputer()
train_X_impute = myimputer.fit_transform(train_X_impute)
val_X_impute = myimputer.fit_transform(val_X_impute)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X_impute, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X_impute)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
print(accuracy_score(val_y,val_predictions))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X_impute, train_y)
val_predictions = iowa_model.predict(val_X_impute)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X_impute, train_y_impute)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


a=RandomForestClassifier(random_state=1,max_depth=100,n_estimators=1000,class_weight='balanced')
a.fit(train_X_impute,train_y)
b=a.predict(val_X_impute)
c=accuracy_score(val_y,b)
print("accuracy ",c)
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the 
rf_model_on_full_data.fit(train_X_impute,train_y_impute)

my_model=XGBRegressor()
my_model.fit(train_X_impute,train_y_impute)


# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]
test_X_impute = test_X.copy()

cols_with_missing = (cols for cols in test_X.columns if test_X[cols].isnull().any())
for cols in cols_with_missing:
    test_X_impute[cols + '_was_missing'] = test_X_impute[cols].isnull()

myimputer = SimpleImputer()
test_X_impute = myimputer.fit_transform(test_X_impute)

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X_impute)
test2=my_model.predict(test_X_impute)
# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test2})

output.to_csv('submission.csv', index=False)
