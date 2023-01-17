# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import re

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes.Date.dtype

earthquakes.Date[0:20]
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# there seems to be three differently formated dates, e.g. '1975-02-23T02:58:41.000Z' from the standard "%m/%d/%Y" format. 
# So for that I will replace it manually first
# do only once
bad_index = [(i,j) for i,j in enumerate(earthquakes.Date) if len(re.findall("\d{2}/\d{2}/\d{4}",j)) == 0]
#bad_index
earthquakes.iloc[3378, 0] = '02/23/1975'
earthquakes.iloc[7512, 0] = '04/28/1985'
earthquakes.iloc[20650, 0] = '03/13/2011'

earthquakes["date_parsed"] = pd.to_datetime(earthquakes["Date"], format= "%m/%d/%Y")

earthquakes.date_parsed.head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquake = earthquakes.date_parsed.dt.day
day_of_month_earthquake.head()
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

# remove na's
day_of_month_earthquake = day_of_month_earthquake.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquake, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)