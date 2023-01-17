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
landslides.head()
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
landslides.sample(5)
# print the first few rows of the date column
print(landslides['date'].head())
# get the number of missing data points per column
missing_values_count = landslides.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:20]
# check the data type of our date column
landslides['date'].dtype
earthquakes.head()
# look at a few rows of the earthquakes file. I can see a handful of missing data already!
earthquakes.sample(5)
# get the number of missing data points per column
missing_values_count = earthquakes.isnull().sum()

# look at the # of missing points in the first twenty columns
missing_values_count[0:20]
# print the first few rows of the Date column
print(earthquakes['Date'].head())
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# inspect the reason
earthquakes.loc[earthquakes["Date"].str.split("/").apply(lambda x:len(x) != 3), "Date"]
# which returns
# 3378     1975-02-23T02:58:41.000Z
# 7512     1985-04-28T02:53:41.530Z
# 20650    2011-03-13T02:23:34.520Z
# Name: Date, dtype: object
# One solution as mentioned above.
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
# instead the
# create a new column, date_parsed, with the parsed dates
# earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%y")
# print the first few rows
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# try to get the day of the month from the Date column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column

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