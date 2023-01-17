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

earthquakes.sample(3)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
print(earthquakes['Date'].dtype)
# (note the capital 'D' in date!)
earthquakes.head(5)

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
landslides.sample(3)
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)
earthquakes.sample(3)
print(earthquakes['date_parsed'].dtype)

# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.sample(5)
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes.sample(5)

# remove na's
import matplotlib.pyplot as plt

day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
fig, ax = plt.subplots(1,3)

sns.distplot(day_of_month_landslides, kde=False, bins=31, ax = ax[0])
ax[0].set_title("Hist")
sns.distplot(day_of_month_landslides, ax = ax[1])
ax[1].set_title("Distr")
sns.distplot(day_of_month_landslides, kde=True, bins=31, ax = ax[2])
ax[2].set_title("Kde")
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

import matplotlib.pyplot as plt

day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
fig, ax = plt.subplots(1,3)

sns.distplot(day_of_month_earthquakes, kde=False, bins=31, ax = ax[0])
ax[0].set_title("Hist")
sns.distplot(day_of_month_earthquakes, ax = ax[1])
ax[1].set_title("Distr")
sns.distplot(day_of_month_earthquakes, kde=True, bins=31, ax = ax[2])
ax[2].set_title("Kde")
volcanos['Last Known Eruption'].sample(5)