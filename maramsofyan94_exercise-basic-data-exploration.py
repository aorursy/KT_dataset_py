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

home_data.describe().transpose()


# What is the average lot size (rounded to nearest integer)?

avg_lot_size = home_data.describe()['LotArea']['mean'].round()

print(avg_lot_size)



# As of today, how old is the newest home (current year - the date in which it was built)

newest_home_age = 2019 - home_data.describe()['YearBuilt']['max']

print(newest_home_age)



# Checks your answers

step_2.check()
import datetime

now = datetime.datetime.now()

year = now.year

# What is the average lot size (rounded to nearest integer)?

avg_lot_size = home_data.describe()['LotArea']['mean'].round()

print(avg_lot_size)



# As of today, how old is the newest home (current year - the date in which it was built)

newest_home_age = year - home_data.describe()['YearBuilt']['max']

print(newest_home_age)



min_area = home_data.describe()['LotFrontage']['min']

print(min_area)



mine_area =home_data['LotFrontage'].min()

print(mine_area)







# Checks your answers

step_2.check()
home_data.describe().LotArea.loc['mean']
home_data.LotArea
home_data.head(8)
home_data.tail()
#step_2.hint()

#step_2.solution()