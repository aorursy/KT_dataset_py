# Code you have previously used to load data

import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex3 import *



print("Setup Complete")
# print the list of columns in the dataset to find the name of the prediction target

home_data.columns
y = home_data.SalePrice



# Check your answer

step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()
# Create the list of features below

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr','TotRmsAbvGrd']



# Select data corresponding to features in feature_names

X = home_data[feature_names]



# Check your answer

step_2.check()
# step_2.hint()

# step_2.solution()
# Review data

# print description or statistics from X

X.describe()



# print the top few lines

X.head()
from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

iowa_model = DecisionTreeRegressor(random_state=1)



# Fit model

iowa_model.fit(X, y)



# Check your answer

step_3.check()
# step_3.hint()

# step_3.solution()
predictions = iowa_model.predict(X)

print(predictions)



# Check your answer

step_4.check()
# step_4.hint()

# step_4.solution()
# You can write code in this cell
