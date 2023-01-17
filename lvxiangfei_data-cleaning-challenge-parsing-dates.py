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
print(earthquakes['Date'])
earthquakes['Date'].dtype
# (note the capital 'D' in date!)

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
earthquakes['Date_parsed']=pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['Date_parsed'].head()

# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
print(day_of_month_landslides)
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquackes=earthquakes['Date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=True, bins=31)
#kde表示是否划线
#bins表示有31个直方
# Your turn! Plot the days of the month from your
day_of_month_earthquackes=day_of_month_earthquackes.dropna()

sns.distplot(day_of_month_earthquackes,kde=False,bins=31)
# earthquake dataset and make sure they make sense.

volcanos['Last Known Eruption'].sample(5)