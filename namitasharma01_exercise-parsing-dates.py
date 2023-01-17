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
earthquakes["Date"].head()
# Check your answer (Run this code cell to receive credit!)
q1.check()
# Line below will give you a hint
q1.hint()
earthquakes[3378:3383]
date_lengths = earthquakes["Date"].str.len()
date_lengths.value_counts()
indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
earthquakes.loc[indices]
# TODO: Your code here
earthquakes.loc[3378, "Date"] = "02/23/1975"
earthquakes.loc[7512, "Date"] = "04/28/1985"
earthquakes.loc[20650, "Date"] = "03/13/2011"

earthquakes["date_parsed"] = pd.to_datetime(earthquakes["Date"], format="%m/%d/%Y")

# Check your answer
q2.check()
# Lines below will give you a hint or solution code
q2.hint()
q2.solution()
# try to get the day of the month from the date column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day

# Check your answer
q3.check()
# Lines below will give you a hint or solution code
q3.hint()
q3.solution()
# TODO: Your code here!

# Check for NA's
day_of_month_earthquakes.isna().sum()

# Plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
# Check your answer (Run this code cell to receive credit!)
q4.check()
# Line below will give you a hint
q4.hint()
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
volcanos.head()
volcanos['Last Known Eruption'].sample(5)
# Number of dates with unknown
volcanos['Last Known Eruption'].value_counts().head()
# Get the indices of dates with unknown
indices = np.where([volcanos['Last Known Eruption'] == "Unknown"])[1]
print('Number of indices with Unknown date:', len(indices))

# Drop the dates with unknown 
parsed_last_known_eruption = volcanos['Last Known Eruption'].drop(index=indices)
# Upper and lower limit on date
print(pd.Timestamp.max, pd.Timestamp.min)
# Split the dates to get the year value seperated from BCE, CE suffix 
parsed_last_known_eruption = parsed_last_known_eruption.str.split().map(lambda x: x[0])

# Pad the dates with 0 on the left for to_datetime to work
parsed_last_known_eruption = parsed_last_known_eruption.str.pad(width=4, fillchar='0')

# Replace out of bounds with NaT
parsed_last_known_eruption = pd.to_datetime(parsed_last_known_eruption, format="%Y", errors="ignore")