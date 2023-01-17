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
landslides[:2]
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
earthquakes[:2]
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
print (landslides['date_parsed'].dtype)
landslides['date_parsed'].head()
#pd.to_datetime(landslides['date_parsed'], infer_datetime_format=True)
earthquakes[:1]
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

#pd.to_datetime(earthquakes['Date'], format = '%m/%d/%y')#infer_datetime_format=True)

### above gave value error : ValueError: unconverted data remains: 65

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
earthquakes[:2]
landslides['date']
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
month_of_landslides = landslides['date_parsed'].dt.month
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31);
# plot the day of the month
sns.distplot(month_of_landslides.dropna(), kde=False, bins=12);
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

sns.distplot(earthquakes['date_parsed'].dt.month, kde=False, bins=12);
volcanos['Last Known Eruption'].sample(5)
pd.to_datetime(volcanos['Last Known Eruption'],  infer_datetime_format=True)