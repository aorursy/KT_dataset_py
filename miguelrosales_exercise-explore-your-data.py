import pandas
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



# Fill in the line below to read the file into a variable home_data

home_data = ____



# Call line below with no argument to check that you've loaded the data correctly

step_1.check()
import pandas as pd

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

home_data.describe()

# Print summary statistics in next line

____
# What is the average lot size (rounded to nearest integer)?

avg_lot_size = ____



# As of today, how old is the newest home (current year - the date in which it was built)

newest_home_age = ____



# Checks your answers

step_2.check()
#step_2.hint()

#step_2.solution()
step_2.hint()
step_2.solution()