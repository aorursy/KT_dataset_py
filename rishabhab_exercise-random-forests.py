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
rf_model = RandomForestRegressor(random_state = 1)

# fit your model
rf_fit_iowa_model=rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y, rf_fit_iowa_model.predict(val_X))

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

# Check your answer
step_1.check()
# The lines below will show you a hint or the solution.
step_1.hint() 
step_1.solution()

from statistics import mean
from matplotlib.pyplot import plot
from seaborn import distplot

#first we will calculate the Decision Tree error values and plot them

DT_errors = val_y - val_predictions
abs_DT_errors = abs(DT_errors)

MAE_DT = mean(abs_DT_errors)

plot(DT_errors, linestyle = ':', c="darkblue")
#Distribution plot

distplot(abs_DT_errors, hist = True, axlabel = 'Absolute Error Distribution')
#Random forest error matrix

RF_errors = val_y - rf_fit_iowa_model.predict(val_X)
abs_RF_errors = abs(RF_errors)

MAE_RF = mean(abs_RF_errors)

plot(RF_errors, linestyle = '--', color = 'Magenta')
distplot(abs_RF_errors, hist= True, color = 'Purple', axlabel = "Absolute Error")