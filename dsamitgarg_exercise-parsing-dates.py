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

earthquakes['Date'].dtype
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

earthquakes.loc[3378, "Date"] = "02/23/1975"

earthquakes.loc[7512, "Date"] = "04/28/1955"

earthquakes.loc[20650, "Date"] = "03/13/2011"

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")

#earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)

# Check your answer

q2.check()
# Lines below will give you a hint or solution code

q2.hint()

#q2.solution()
# try to get the day of the month from the date column

day_of_month_earthquakes = earthquakes['date_parsed'].dt.day



# Check your answer

q3.check()
# Lines below will give you a hint or solution code

#q3.hint()

#q3.solution()
# TODO: Your code here!

day_of_month_earthquakes = day_of_month_earthquakes.dropna()

sns.distplot(day_of_month_earthquakes, kde = False, bins = 31)
# Check your answer (Run this code cell to receive credit!)

q4.check()
# Line below will give you a hint

#q4.hint()
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
volcanos['Last Known Eruption'].sample(5)