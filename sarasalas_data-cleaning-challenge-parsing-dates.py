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
earthquakes['Date'].dtype
print(earthquakes['Date'])
# (note the capital 'D' in date!)

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
#landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
#earthquakes['eq_date_parsed'] = pd.to_datetime(earthquakes['Date'],format="%m/%d/%Y")
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'],infer_datetime_format=True)

earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
#I comment this code because it returns an error
#day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquake = earthquakes['date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquake = day_of_month_earthquake.dropna()
sns.distplot(day_of_month_earthquake, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)