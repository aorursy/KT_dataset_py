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
print(earthquakes.shape)
print(earthquakes.columns)
earthquakes['Date'].dtypes
# Lets see the data :: We have some inconsistent data in Date column at  rows number 3378, 7512, and 20650. 
print (earthquakes.shape)

earthquakes.sort_values('Date',ascending=False).head()
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")
# Subset rows 3378, 7512, and 20650

junk_earthquakes_ds = earthquakes.iloc[[3378,7512,20650]]
junk_earthquakes_ds.head()
# Subset rows rows 3378, 7512, and 20650

earthquakes_ds = earthquakes.drop([3378,7512,20650], axis=0)
earthquakes_ds.sort_values('Date',ascending=False).head()
# create a new column, date_parsed, with the parsed dates
earthquakes_ds['date_parsed'] = pd.to_datetime(earthquakes_ds['Date'], format = "%m/%d/%Y")
earthquakes_ds.head()
pd.options.mode.chained_assignment = None
junk_earthquakes_ds['date_parsed'] = pd.to_datetime(junk_earthquakes_ds['Date'],format = "%Y-%m-%d")
junk_earthquakes_ds.head()
print(junk_earthquakes_ds.shape)
print(earthquakes_ds.shape)
earthquakes_new = earthquakes_ds.append(junk_earthquakes_ds,ignore_index = True)
print(earthquakes_new.shape)
earthquakes_new.tail()
del earthquakes_ds
del junk_earthquakes_ds
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes_new['date_parsed'].dt.day
print(earthquakes_new['date_parsed'].head(),day_of_month_earthquakes.head())
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