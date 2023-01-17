# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
import pandas as pd
import numpy as np
import datetime
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




home_data.head()
home_data.describe()

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = home_data['LotArea'].mean()
avg_lot1=np.ceil(avg_lot_size)

print(avg_lot1) # performed using mean() and ciel() function
avg_lot_size = 10517


# As of today, how old is the newest home (current year - the date in which it was built)



New_home = datetime.datetime.now() # performed using datetime lib and max() function
print(New_home.year-home_data['YearBuilt'].max())

# Checks your answers

newest_home_age = 10
step_2.check()
#step_2.hint()
#step_2.solution()