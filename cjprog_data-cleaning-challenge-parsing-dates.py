# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)
earthquakes.head()
landslides.head()
volcanos.head()
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
print(earthquakes['Date'].head())

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# This works fine:
pd.to_datetime(earthquakes['Date'].head(), format = "%m/%d/%Y")

# but this trows a ValueError: 
# time data '1975-02-23T02:58:41.000Z' does not match format '%m/%d/%Y' (match).
#earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")

# so we should better use the infer method.
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)

# show the first rows to see if 'date_parsed' has been added.
earthquakes.head()
# double-check that the dtype is correct.
earthquakes['date_parsed'].head()
earthquakes['date_parsed'].dtype
np.dtype('datetime64[ns]') == np.dtype('<M8[ns]')
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
pd.concat([day_of_month_landslides.head(), landslides['date_parsed'].head()], keys=['day_of_month_landslides', 'date_parsed'], axis=1)
# Your turn! get the day of the month from the date_parsed column in the earthquakes dataset
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
pd.concat([day_of_month_earthquakes.head(), earthquakes['date_parsed'].head()], keys=['day_of_month_earthquakes', 'date_parsed'], axis=1)

# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquakes dataset and make sure they make sense.

# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)

volcanos['Last Known Eruption'].sample(5)