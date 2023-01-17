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

# print the first few rows of the date column
print(earthquakes['Date'].head())
# check the data type of our date column
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# create a new column, date_parsed, with the parsed dates
earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], errors = 'coerce', format = "%m/%d/%Y")
# I observed that running the command:
# earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")
# Gives an error that translates to the fact that there are cells that do not agree with the format="%m/%d/%Y" 
# because they're in the format="%d/%m/%Y"
# So I added errors = 'coerce' to ignore the error and go ahead and parse the date

earthquakes['Date_parsed'].head()
# I suspect that some values of 'Date' were not converted correctly, which is why I coerced the errors in the line above. 
# Coerced errors are converted to the 'NaT' which is a null value. Let's see if null values exist in the new variable.

missing_dates = earthquakes[earthquakes.Date_parsed.isnull()] 
print(missing_dates.shape)

# There are 3 missing date values. This is because the dates for these 3 observations are in a different format. 
# For those 3 observations, let Python infer the date format.

earthquakes.loc[earthquakes.Date_parsed.isnull() == True, 'Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)


# Now lets see how many missing values we have.

missing_dates = earthquakes[earthquakes.Date_parsed.isnull()] 
print(missing_dates)

# Success! No more missing dates.
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
print(day_of_month_landslides.head())
# Also get the month from the date_parsed column
month_landslides = landslides['date_parsed'].dt.month
print(month_landslides.head())
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['Date_parsed'].dt.day
print(day_of_month_earthquakes.head())
# Also get the month from the date_parsed column
month_earthquakes = earthquakes['Date_parsed'].dt.month
print(month_earthquakes.head())
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

# remove na's
# day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)