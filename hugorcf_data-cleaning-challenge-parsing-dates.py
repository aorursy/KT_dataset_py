# modules we'll use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# check first entries
print(earthquakes['Date'].head(3))

# create new column data_parsed
# option erros='coerce': any invalid parsing will be set as NaT
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y", errors='coerce')
# check if any dates were set as NaT
missing_dates = earthquakes[earthquakes['date_parsed'].isnull()] 
print(missing_dates.shape)
# for the 3 NaT, use infer_datetime_format=True
earthquakes.loc[earthquakes.date_parsed.isnull() == True, 'date_parsed'] = \
pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
# check first entries of data_parsed
print(earthquakes['date_parsed'].head(3))
print(earthquakes['date_parsed'].dtype)
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# check first few values
day_of_month_earthquakes.head(5)
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
volcanos['Last Known Eruption'].head(7)
# TEMPORARY - TO DELETE!!!!!!!!!!!!!!!!!!!!!!!!!!!!
volcanos.drop(columns=['date_parsed'], inplace=True)
volcanos['date_parsed'] = volcanos['Last Known Eruption']
# delete ' CE' from entries
volcanos['date_parsed'].replace({' CE':''}, regex=True, inplace=True)
# add a '-' to entries with ' BCE' and then delete ' BCE' from entries
volcanos.loc[volcanos['date_parsed'].str.contains(' BCE'), 'date_parsed'] = ("-"+volcanos['date_parsed'])
volcanos.loc[volcanos['date_parsed'].str.contains(' BCE'), 'date_parsed'] = volcanos['date_parsed'].replace({' BCE':''}, regex=True, inplace=True)
# replace 'Unknown' with NaN
volcanos['date_parsed'].replace('Unknown', np.NaN, inplace=True)
# convert non NaN entries to int64
volcanos['date_parsed'].dropna(inplace=True)
volcanos['date_parsed'] = volcanos['date_parsed'].astype(np.int64, copy=False)
# check a few entries of data_parsed
print(volcanos['date_parsed'].sample(5))
# remove na's
years_volcanos = volcanos['date_parsed'].dropna()

# plot a distplot with the years of volcano eruptions
sns.distplot(years_volcanos, kde=False, bins=31)
_ = plt.xlabel('year')
_ = plt.title('Volcano eruptions')