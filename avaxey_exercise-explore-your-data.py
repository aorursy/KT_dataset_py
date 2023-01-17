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

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print summary statistics in next line

home_data.describe()
# What is the average lot size (rounded to nearest integer)?

avg_lot_size = round(home_data.LotArea.mean())



# As of today, how old is the newest home (current year - the date in which it was built)

newest_home_age = 2019 - home_data.YearBuilt.max()



# Checks your answers

step_2.check()
#step_2.hint()

#step_2.solution()
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path) 

melbourne_data.columns

y = melbourne_data.Price

y.describe()
melbourne_features = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude']

X = melbourne_data [melbourne_features]

X.describe()
from sklearn.tree import DecisionTreeRegressor



melbourne_model = DecisionTreeRegressor(random_state=10)

melbourne_model.fit(X,y)
print("Making predictions for the following 5 houses")

print(X.head)

print("The predictions are")

print(melbourne_model.predict(X.head()))