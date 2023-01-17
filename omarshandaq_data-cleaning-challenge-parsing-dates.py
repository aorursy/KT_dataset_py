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
print(earthquakes['Date'].head())
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

# NOTE: we must to get the right date format first then we can format it to "%m/%d/%Y" and the "Y" is capital because 
# the date in earthquakes dataset have the year in four digits.
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True, format = "%m/%d/%Y")
earthquakes['date_parsed'].head()
# also the data of "date_parsed" will be datetime64[ns]

# try to get the day of the month from the date column
# day_of_month_landslides = landslides['date'].dt.day
# NOTE: there are error here because the dt just work with datetimelike and then you put the value.
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
# first of we need to remove na's values from "day_of_month_earthqukaes" so the plot goes right even 
# though it when the same but just in case it has any na's values.

day_of_month_earthquakes = day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)

# I see that it makes sense because the dataset "earthquakes" have alot more dates than "landslides" the "earthquakes" dataset have more than 
# 23000 dates while "landslides" dateset have less that 4000, thats why the plot of "earthquakes" jumped over 800 number of days while 
# "landsliedes just went to 80 number of days.
volcanos['Last Known Eruption'].sample(5)