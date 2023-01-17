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

earthquakes.Date.dtype
# Check your answer (Run this code cell to receive credit!)

q1.check()
# Line below will give you a hint

#q1.hint()
earthquakes[3378:3383]
date_lengths = earthquakes.Date.str.len()

date_lengths.value_counts()
indices = np.where([date_lengths == 24])[1]

print(indices)



print('Indices with corrupted data:', indices)

earthquakes.loc[indices]
earthquakes.loc[earthquakes['Date'].str.len() == 24, 'Date'] = '02/23/1975'

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format='%m/%d/%Y')

# Check your answer

q2.check()
# Lines below will give you a hint or solution code

#q2.hint()

q2.solution()
# try to get the day of the month from the date column

day_of_month_earthquakes = earthquakes.date_parsed.dt.day



# Check your answer

q3.check()
# Lines below will give you a hint or solution code

#q3.hint()

#q3.solution()
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
# Check your answer (Run this code cell to receive credit!)

q4.check()
# Line below will give you a hint

#q4.hint()
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
volcanos['Last Known Eruption'].sample(5)

unknown_eruption_data = volcanos.loc[volcanos['Last Known Eruption'] == 'Unknown']

#print(unknown_eruption_data)

volcanos_known_eruption =  volcanos.loc[volcanos['Last Known Eruption'] != 'Unknown']

known_eruption_data = volcanos_known_eruption['Last Known Eruption']

eruption_year =  known_eruption_data.str[0:4]

eruption_era = np.where((known_eruption_data.str[-3:] == 'BCE'),'BC','AD')

volcanos_known_eruption['eruption_year'] = eruption_year.map(lambda x: x.strip())

volcanos_known_eruption['eruption_era'] = eruption_era



#print(volcanos_known_eruption['eruption_era'])



volcanos_known_eruption[['eruption_year', 'eruption_era']].value_counts().sort_index().plot.line(subplots=True)