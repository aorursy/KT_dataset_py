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
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
index_good_date = ~earthquakes['Date'].str.contains('T')
index_bad_dates = earthquakes['Date'].str.contains('T')
bad_date_earthq = earthquakes.Date.loc[index_bad_dates]
good_date_earthq = earthquakes.loc[index_good_date]
earthquakes['date_format'] = pd.to_datetime(good_date_earthq['Date'], format = "%m/%d/%Y")
earthquakes.at[3378, 'date_format']=pd.to_datetime('1975-02-23', format = "%Y-%m-%d") 
earthquakes.at[7512, 'date_format']=pd.to_datetime('1985-04-28', format = "%Y-%m-%d") 
earthquakes.at[20650, 'date_format']=pd.to_datetime('2011-03-13', format = "%Y-%m-%d") 

print(earthquakes.date_format.iloc[3376])
print(earthquakes.date_format.iloc[3377])
print(earthquakes.date_format.iloc[3378])
print(earthquakes.date_format.iloc[3379])
#

# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.sample()
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_format'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes, bins=31, kde = False)
volcanos['Last Known Eruption'].sample(5)