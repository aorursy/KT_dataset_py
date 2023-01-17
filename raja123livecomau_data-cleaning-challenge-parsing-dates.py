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
landslides.head(5)

# get the number of missing data points per column
missing_values_count = landslides.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
# how many total missing values do we have?
total_cells = np.product(landslides.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# remove all columns with at least one missing value
columns_with_na_dropped = landslides.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % landslides.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
landslides.fillna(method = 'bfill', axis=0).fillna(0)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)landslides['date'].dtype
earthquakes.head(5)


# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
earthquakes.fillna(method = 'bfill', axis=0).fillna(0)
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)

earthquakes['Date_parsed'].dtype
earthquakes['Date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['Date_parsed'].dt.day
day_of_month_earthquakes
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)