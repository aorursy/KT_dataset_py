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
landslides.dtypes
landslides.get_dtype_counts()
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)

earthquakes.Date.sample(5)
earthquakes.Date.dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
earthquakes['Date_parsed'].head()
earthquakes['Date'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
landslides['date_parsed'].head()
# Your turn! get the day of the month from the date_parsed column

day_of_month_earthquakes = earthquakes['Date_parsed'].dt.day
day_of_month_earthquakes.head()
earthquakes['Date_parsed'].head()
day_of_month_landslides.loc[day_of_month_landslides.isnull()]
landslides['date'].loc[[1482, 1497, 1498]]
landslides['date_parsed'].loc[[1482, 1497, 1498]]
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

day_of_month_earthquakes.isnull().sum()
# remove na's (but since there are no na's, this step is not useful here)
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
sns.distplot(day_of_month_earthquakes, bins=31)
volcanos['Last Known Eruption'].sample(5)
volcanos['Last Known Eruption'].head(5)
volcanos['Last Known Eruption'].dtype
volcanos['Last Known Eruption'].loc[3][:-4]
# getting to know the unkown count
a = volcanos['Last Known Eruption'] == "Unknown"
print(a.sum())

# Uncomment if you want to remove the unkown rows

# unknown_idxs = volcanos.index[volcanos['Last Known Eruption'] == "Unknown"]
# volcanos2 = volcanos.drop(unknown_idxs)

# a = volcanos2['Last Known Eruption'] == "Unknown"
# print a.sum()
# Fix the Last Known Eruption values
volcanos['Last Known Eruption parsed'] = volcanos['Last Known Eruption'].apply(lambda x: -int(x[:-4]) if x.endswith('BCE') else (int(x[:-3]) if x.endswith('CE') else np.nan))
# Parse the dates
volcanos['Last Known Eruption parsed'] = volcanos['Last Known Eruption parsed'].astype('float64')
print(volcanos['Last Known Eruption'].head(), '\n')
print(volcanos['Last Known Eruption parsed'].head())
# plotting years after dropping na's
sns.distplot(volcanos['Last Known Eruption parsed'].dropna(), kde=False)
# Zomming in the graph

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_ylim(0, 50)
sns.distplot(volcanos['Last Known Eruption parsed'].dropna(), kde=False, ax=ax)
