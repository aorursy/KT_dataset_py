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
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# There are two rows in the dataset that doesn't match the %m/%d/%Y. 
# Instead, their patterns are "%Y-%m-%d", e.g. 1975-02-23T02:58:41.000Z
earthquakes['lenDate'] = earthquakes['Date'].apply(len)
print("Here are the abnormal rows:")
print(earthquakes.loc[earthquakes['lenDate'] > 10]['Date'])

# There are two solutions to this problem
# Solution 1: Remove the rows that doesn't match the patterm by examine the length of the Date field
earthquakes_with_normal_dates = earthquakes.loc[earthquakes['lenDate'] < 11].copy()
earthquakes_with_normal_dates['Date_parsed'] = pd.to_datetime(earthquakes_with_normal_dates['Date'],
                                                             format = "%m/%d/%Y")
print("Solution 1")
print(earthquakes_with_normal_dates['Date_parsed'].head())
# Solution 2: Use pandas built-in infer functionality
earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
print("Solution 2")
print(earthquakes['Date_parsed'].head())
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['Date_parsed'].dt.day
day_of_month_earthquakes.head()
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