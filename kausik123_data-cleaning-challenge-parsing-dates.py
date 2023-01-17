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
print(earthquakes.Date.head())
print(earthquakes.Date.dtype)
# (note the capital 'D' in date!)
# Date is object here too. We are seeing this as Date format but Python doesn't, Python sees as string.
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
earthquakes[earthquakes.Date == '02/23/1975']
# Your turn! Create a new column, date_parsed, in the earthquakes

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")

# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
# So looks like length is more than 10 for some for the cases. Let's find out.
earthquakes['DateLen'] = earthquakes.Date.apply(lambda x: len(x))
earthquakes[earthquakes.DateLen > 10]
cleaned_earthquakes = earthquakes[earthquakes.DateLen < 11]
cleaned_earthquakes[cleaned_earthquakes.DateLen > 10]
cleaned_earthquakes['date_parsed'] = pd.to_datetime(cleaned_earthquakes['Date'], format = "%m/%d/%Y")
cleaned_earthquakes.date_parsed.head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = cleaned_earthquakes['date_parsed'].dt.day
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