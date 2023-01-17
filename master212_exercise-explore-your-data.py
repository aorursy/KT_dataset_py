# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



# Fill in the line below to read the file into a variable home_data

home_data = melbourne_data = pd.read_csv(iowa_file_path )

#melbourne_data.describe()



# Call line below with no argument to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print summary statistics in next line

melbourne_data.describe()

# What is the average lot size (rounded to nearest integer)?

import datetime

now = datetime.datetime.now()

avg_lot_size =  int(round(home_data['LotArea'].mean()))

print(avg_lot_size )



# As of today, how old is the newest home (current year - the date in which it was built)

newest_home_age = now.year - (int(round(home_data['YearBuilt'].max())))

print(newest_home_age)



# Checks your answers

step_2.check()
# print(home_data.describe)

x = round(home_data.describe().LotArea.loc['mean'])

y = 2019 - home_data.describe().YearBuilt.loc['max'] +2000

print(x, "     ",y)

step_2.hint()

#step_2.solution()