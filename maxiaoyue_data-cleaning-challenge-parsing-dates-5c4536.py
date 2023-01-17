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
print ('earthquakes', earthquakes.shape)
print ('landslides', landslides.shape)
print ('volcanos', volcanos.shape)
earthquakes.dtypes
earthquakes.head()
landslides.head()
volcanos.head()
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes.Date.dtype
earthquakes.Date.head()
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes.Date, infer_datetime_format=True)
print (earthquakes.date_parsed.dtype)
earthquakes.date_parsed.head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()

# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes.date_parsed.dt.day
day_of_month_earthquakes.head()
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
day_of_month_earthquakes.isnull().sum()
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
sns.distplot(day_of_month_earthquakes, kde = False, bins = 31)
volcanos['Last Known Eruption'].sample(15)
volcanos['Last Known Eruption'].describe()
volcanos['Last Known Eruption'].isnull().sum()
year_of_eruption = volcanos['Last Known Eruption']
year_of_eruption.isnull().sum()
year_of_eruption_unknownToNone = year_of_eruption.replace({'Unknown': None},)
year_of_eruption_unknownToNone.isnull().sum()
year_of_eruption_df = year_of_eruption_unknownToNone.str.split(' ', expand = True)
year_of_eruption_df.columns = ['year', 'era']
year_of_eruption_df.head()
year_of_eruption_df['year'] = pd.to_numeric(year_of_eruption_df.year)
year_of_eruption_df.head()
year_of_eruption_df['era'] = year_of_eruption_df['era'].replace({'BCE': -1, 'CE': 1})
year_of_eruption_df['era'].head()
year_of_eruption_final = year_of_eruption_df.year * year_of_eruption_df.era
year_of_eruption_final.head()
sns.distplot(year_of_eruption_final.dropna(), kde = False)
volcanos['year'] = pd.to_numeric(year_of_eruption.apply(lambda x: -int(x.split()[0]) if x.endswith('BCE') else int(x.split()[0]) if x.endswith('CE') else None))
volcanos['year'].head()
sns.distplot(volcanos['year'].dropna(), kde = False)