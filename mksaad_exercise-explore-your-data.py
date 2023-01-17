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
home_data
home_data.columns
home_data.LotArea
home_data.LotArea.mean()
home_data.describe().LotArea
home_data.describe().LotArea.loc['mean']
home_data.describe().transpose()
home_data.tail()
round(home_data.describe().LotArea.loc['mean'])
round(home_data.LotArea.mean())
import datetime

now = datetime.datetime.now()

print(now)

print(now.year)

type(now.year)
# What is the average lot size (rounded to nearest integer)?

avg_lot_size = round(home_data.describe().LotArea.loc['mean'])



# As of today, how old is the newest home (current year - the date in which it was built)

import datetime

now = datetime.datetime.now()



newest_home_age = now.year - home_data.describe().YearBuilt.loc['max']



# Checks your answers

step_2.check()
#step_2.hint()

#step_2.solution()