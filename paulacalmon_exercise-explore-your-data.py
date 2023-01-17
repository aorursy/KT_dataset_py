# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



# Fill in the line below to read the file into a variable home_data

home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv')



# Call line below with no argument to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print summary statistics in next line

print(home_data.describe())

print(home_data.dtypes)
avg_lot_size = home_data["LotArea"].mean()

newest_home = home_data["YearBuilt"].max()

print(int(avg_lot_size))

print(newest_home)
#step_2.hint()

#step_2.solution()