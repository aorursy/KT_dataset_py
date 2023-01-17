# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



# Fill in the line below to read the file into a variable home_data

home_data = pd.read_csv(iowa_file_path)



# Call line below with no argument to check that you've loaded the data correctly

print(home_data)
home_data = pd.read_csv(iowa_file_path)
# Print summary statistics in next line

print(home_data.describe())
home_data.columns
# What is the average lot size (rounded to nearest integer)?

avg_lot_size =home_data.mean(axis=0)



# As of today, how old is the newest home (current year - the date in which it was built)

newest_home_age = home_data['YearBuilt'] - home_data['YearRemodAdd']



y = home_data.SalePrice
home_features = ['LotArea', 'OverallCond', 'YearBuilt','MoSold']
X = home_data[home_features]
X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

melbourne_model = DecisionTreeRegressor(random_state=1)



# Fit model

melbourne_model.fit(X, y)
print("Making predictions for the following houses:")

print(X.head())

print("The predictions are")

print(melbourne_model.predict(X.head()))