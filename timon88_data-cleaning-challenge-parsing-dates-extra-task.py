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
print(earthquakes.Date.head())
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
earthquakes['Date_parsed'].head()
earthquakes.tail()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['Date_parsed'].dt.day
day_of_month_earthquakes.describe()
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
volcanos['Last Known Eruption'] = volcanos['Last Known Eruption'].apply(lambda x: np.nan if x == 'Unknown' else x)
volcanos['Last Known Eruption Year'] = volcanos['Last Known Eruption'].apply(lambda x: np.nan if pd.isnull(x) else str(x).split(' ')[0])
volcanos['Last Known Eruption BCE/CE'] = volcanos['Last Known Eruption'].apply(lambda x: 'BCE' if str(x)[-3] == 'B' else ('CE' if str(x)[-3] == ' ' else np.nan))
volcanos.head()
def years_from_2018(row):
    if pd.isnull(row['Last Known Eruption Year']):
        return np.nan
    elif row['Last Known Eruption BCE/CE'] == 'CE':
        return 2018 - int(row['Last Known Eruption Year'])
    else:
        return 2018 + int(row['Last Known Eruption Year'])
volcanos['Years from 2018'] = volcanos.apply(years_from_2018, axis = 1)
volcanos.head()
volcanos.describe()