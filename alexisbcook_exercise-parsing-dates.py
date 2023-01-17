from learntools.core import binder

binder.bind(globals())

from learntools.data_cleaning.ex3 import *

print("Setup Complete")
# modules we'll use

import pandas as pd

import numpy as np

import seaborn as sns

import datetime



# read in our data

earthquakes = pd.read_csv("../input/earthquake-database/database.csv")



# set seed for reproducibility

np.random.seed(0)
# TODO: Your code here!

# Check your answer (Run this code cell to receive credit!)

q1.check()
# Line below will give you a hint

#q1.hint()
earthquakes[3378:3383]
date_lengths = earthquakes.Date.str.len()

date_lengths.value_counts()
indices = np.where([date_lengths == 24])[1]

print('Indices with corrupted data:', indices)

earthquakes.loc[indices]
# TODO: Your code here



# Check your answer

q2.check()
# Lines below will give you a hint or solution code

#q2.hint()

#q2.solution()
# try to get the day of the month from the date column

day_of_month_earthquakes = ____



# Check your answer

q3.check()
# Lines below will give you a hint or solution code

#q3.hint()

#q3.solution()
# TODO: Your code here!

# Check your answer (Run this code cell to receive credit!)

q4.check()
# Line below will give you a hint

#q4.hint()
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
volcanos['Last Known Eruption'].sample(5)