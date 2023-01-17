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
home_data = pd.read_csv(iowa_file_path)

home_data
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print summary statistics in next line

home_data.describe()

# Here Keys(Name Index) or Staticses is Name Of Raws While The Columns Is The Name Of Antributes
home_data.describe().transpose()

# Here Columns is Name Of Staticses While The Raws Is The Name Of Antributes
home_data.LotArea
# name of data.name of antribute.name of static(name of raw)

# easier way :-

home_data.LotArea.mean() # To Find The Value Of Average(mean) For The antribute LotArea Without Calculate It 
# here We Get The Value Of Average For The antribute LotArea by calculating

# harder way and will take a lot of time because will research in all raws and columns

home_data.describe().LotArea.loc['mean']
# To Find Specific Column From The Describe Of DataFrame

home_data.describe().LotArea
avg_lot_size = round(home_data.LotArea.mean())

print(avg_lot_size)
# What is the average lot size (rounded to nearest integer)?

avg_lot_size = round(home_data.LotArea.mean())



# As of today, how old is the newest home (current year - the date in which it was built)

import datetime

#current year = datetime.date.now().year

newest_home_age = datetime.datetime.now().year - home_data.YearBuilt.max()



# Checks your answers

step_2.check()
now = datetime.datetime.now() # Will Print THe Time Of Server Not My Computer

print(now)
#step_2.hint()

#step_2.solution()