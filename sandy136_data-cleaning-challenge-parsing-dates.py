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
# check earthquakes data
earthquakes.head(30)
# check out detail information about each columns in earthquakes data
earthquakes.info()
# check landslides data
landslides.head(30)
# check out detail information about each columns in landslides data
landslides.info()
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
# print the first 10 rows of the Date column in the earthquakes dataframe
print(earthquakes['Date'].head(10))
# check the data type of our date column
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
## check the data type of date parsed column
landslides['date_parsed'].dtype

# There is no difference between np.dtype('datetime64[ns]') and np.dtype('<M8[ns]')
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
# this code will occur ValueError : time data '1975-02-23T02:58:41.000Z' does not match format '%m/%d/%Y' (match)
# We can assume that the time data format is not unified
earthquakes['date_parsed']=pd.to_datetime(earthquakes['Date'],format="%m/%d/%Y")
# to solve the ValueError we will be using errors parameter in to_datetime function
# to_datetime(arg, *errors, dayfirst, yearfirst, utc, box, format, exact, unit, infer_datetime_format, origin)
# if there is an error, error='coerce' will automatically replace error into NaT value
earthquakes['date_parsed']=pd.to_datetime(earthquakes['Date'],format="%m/%d/%Y",errors='coerce')

# NaT is checked by pd.isnull()
pd.isnull(np.datetime64('NaT'))
# print the first few rows
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# let's check out day of month landslides data
day_of_month_landslides.head(10)
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# let's check out day of month landslides data
day_of_month_earthquakes.head(10)
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)