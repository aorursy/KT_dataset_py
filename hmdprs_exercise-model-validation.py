# load data

import pandas as pd

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)



# select the prediction target

y = home_data['SalePrice']



# choose features

feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_columns]



# specify model

from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor()



# fit model

iowa_model.fit(X, y)



print("First in-sample predictions:", iowa_model.predict(X.head()))

print("Actual target values for those homes:", y.head().tolist())



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex4 import *

print("Setup Complete")
# import the train_test_split function and uncomment

from sklearn.model_selection import train_test_split



# split data to train and validation sets

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Check your answer

step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()
# You imported DecisionTreeRegressor in your last exercise

# and that code has been copied to the setup code above. So, no need to

# import it again



# specify the model

iowa_model = DecisionTreeRegressor(random_state=1)



# fit iowa_model with the training data.

iowa_model.fit(train_X, train_y)



# Check your answer

step_2.check()
# step_2.hint()

# step_2.solution()
# Predict with all validation observations

val_predictions = iowa_model.predict(val_X)



# Check your answer

step_3.check()
# step_3.hint()

# step_3.solution()
# print the top few validation predictions

print('Prediction:', val_predictions[:5])

# print the top few actual prices from validation data

print('Actual:', val_y.head().tolist())
# mae in validation data

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)



# uncomment following line to see the validation_mae

print(val_mae)



# Check your answer

step_4.check()
# step_4.hint()

# step_4.solution()