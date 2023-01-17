# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
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


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex6 import *
print("\nSetup complete")
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state =1)

# fit your model
rf_model.fit(train_X,train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
val_pred = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_pred,val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

missing_val_count_by_column = (home_data.isnull().sum())
#print(missing_val_count_by_column)
print("no of columns" , home_data.columns.size)
#missining_val_drop_data = home_data.dropna(axis =1)
#print("after removing missing" ,missining_val_drop_data.columns.size)
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_train_X = my_imputer.fit_transform(train_X)
imputed_val_X = my_imputer.transform(val_X)
rf_model.fit(imputed_train_X,train_y)
val_pred = rf_model.predict(imputed_val_X)
rf_val_mae = mean_absolute_error(val_pred,val_y)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

#print(home_data.dtypes.sample(10))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(),RandomForestRegressor())
from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline,X,y,scoring = 'neg_mean_absolute_error')
print("cross validation", -1 * scores.mean())

step_1.check()
# The lines below will show you a hint or the solution.
step_1.hint() 
step_1.solution()
