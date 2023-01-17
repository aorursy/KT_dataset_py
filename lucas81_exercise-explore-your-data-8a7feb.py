# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



# Fill in the line below to read the file into a variable home_data

home_data = _pd.read_csv('iowa_file_path')

home_data



# Call line below with no argument to check that you've loaded the data correctly

step_1.check(home_data)
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print summary statistics in next line

home_data.describe()
# What is the average lot size (rounded to nearest integer)?

avg_lot_size =average.round(lot_size)

avg_lot_size



# As of today, how old is the newest home (current year - the date in which it was built)

newest_home_age = current_year - buit_date

newest_home_age



# Checks your answers

step_2.check(newest_home_age)
#step_2.hint()

#step_2.solution()