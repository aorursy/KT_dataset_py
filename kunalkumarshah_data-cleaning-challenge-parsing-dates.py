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
print(earthquakes['Date'].head())
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# find the length of the Date Column
earthquakes['lenDate'] = earthquakes['Date'].apply(len)
earthquakes.loc[earthquakes['lenDate'] > 10]

# START : convert the normal dates to DateTime
# All normal dates are with length less than 11 having format mm/dd/yyyy
earthquakes_with_normal_dates = earthquakes.loc[earthquakes['lenDate'] < 11]

# since in the earlier step we made a slice of the actual dataframe, we will copy it , so that slice of normal dates itself becomes a dataframe
earthquakes_with_normal_dates = earthquakes_with_normal_dates.copy()

# convert to DateTime and verify the data type 
earthquakes_with_normal_dates['date_parsed'] = pd.to_datetime(earthquakes_with_normal_dates['Date'], format = "%m/%d/%Y")
print(earthquakes_with_normal_dates['date_parsed'].head())
# END : convert the normal dates to DateTime

# START : convert the odd dates to DateTime
# All odd dates are with length > 11 having format yyyy-mm-dd
earthquakes_with_odd_dates = earthquakes.loc[earthquakes['lenDate'] > 11]
# since in the earlier step we made a slice of the actual dataframe, we will copy it , so that slice of normal dates itself becomes a dataframe
earthquakes_with_odd_dates = earthquakes_with_odd_dates.copy()
# convert to DateTime 
earthquakes_with_odd_dates['date_parsed'] = pd.to_datetime(earthquakes_with_odd_dates['Date'], format = "%Y-%m-%d")
# remove the Time from the datetime column and verify the data type 
earthquakes_with_odd_dates['date_parsed'] = earthquakes_with_odd_dates['date_parsed'].dt.date
earthquakes_with_odd_dates['date_parsed'] = pd.to_datetime(earthquakes_with_odd_dates['date_parsed'], format = "%Y-%m-%d")
print(earthquakes_with_odd_dates['date_parsed'].head())
# END : convert the odd dates to DateTime

# concat the normal and odd dates dataframe and verify that it has the same number of rows as the original earthquakes dataframe
earthquakes_new = pd.concat([earthquakes_with_normal_dates,earthquakes_with_odd_dates])
print(earthquakes.shape)
print(earthquakes_with_normal_dates.shape)
print(earthquakes_with_odd_dates.shape)
print(earthquakes_new.shape)

# Simpler way instead of breaking dataframe into normal and odd dates, let Pandas infer what dateformat is and let it convert
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
print(earthquakes['date_parsed'].head())
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
print(day_of_month_earthquakes)
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)