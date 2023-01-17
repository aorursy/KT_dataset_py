# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)
landslides.date.sample(5)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes.Date.sample(15)

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes["date_parsed"] = pd.to_datetime(earthquakes.Date, infer_datetime_format=True) # format="%m/%d/%Y")
earthquakes.date_parsed.sample(5)
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides.date_parsed.dt.day # landslides['date_parsed'].dt.day
# day_of_month_landslides
# Your turn! get the day of the month from the date_parsed column
day_of_earthquakes = earthquakes.date_parsed.dt.day
# day_of_earthquakes
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
fig, ax=plt.subplots(1, 2)
sns.distplot(day_of_month_landslides, bins=31, ax=ax[0])
sns.distplot(day_of_month_landslides, kde=False, bins=31, ax=ax[1])
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

earthquakes_days = earthquakes.date_parsed.dt.day.dropna()

fig, ax=plt.subplots(1,2)
sns.distplot(earthquakes_days, bins=31, ax=ax[0])
ax[0].set_title("As distribution")
sns.distplot(earthquakes_days, bins=31, kde=False, ax=ax[1])
ax[1].set_title("Just values")
volcanos['Last Known Eruption'].sample(5)