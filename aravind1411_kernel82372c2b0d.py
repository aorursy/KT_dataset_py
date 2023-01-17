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
home_data.head()
# Lines below will give you a hint or solution code
#step_1.hint()
step_1.solution()
# Print summary statistics in next line
home_data.describe()
#print(home_data.columns)
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = home_data['LotArea'].round().astype(int).mean()
print(round(avg_lot_size))
import datetime
curyear=datetime.datetime.now()
# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = curyear.year-home_data['YrSold']
print(newest_home_age.min())

# Checks your answers

#step_2.hint()
step_2.solution()
