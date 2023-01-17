# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd



# Path of the file to read

iowa_file_path = pd.read_csv('../input/home-data-for-ml-course/train.csv')



# Fill in the line below to read the file into a variable home_data

home_data = pd.DataFrame(iowa_file_path)



# Call line below with no argument to check that you've loaded the data correctly

home_data.describe()
# Lines below will give you a hint or solution code

home_data.describe()

# home_data.solution()
# Print summary statistics in next line

home_data.describe()

print(home_data.YearBuilt.head(100))
import numpy as np



# What is the average lot size (rounded to nearest integer)?

avg_lot_size = home_data.LotArea.mean()

print(avg_lot_size)



import datetime

val_date = pd.to_datetime('1-08-2019')

home_data['YearBuilt'] = pd.to_datetime(home_data['YearBuilt'])

print(val_date)



# As of today, how old is the newest home (current year - the date in which it was built)

# print(val_date - home_data.YearBuilt)

newest_home_age = (val_date - home_data.YearBuilt)/np.timedelta64(1,'Y')



# Checks your answers

print(newest_home_age)

#print(va_date) 
#step_2.hint()

#step_2.solution()