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
earthquakes.sample(5)
landslides.sample(5)
volcanos.sample(5)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype
earthquakes['Date'].sample(5)
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
landslides['date_parsed'].loc[landslides['date_parsed'].isnull()]
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y", errors='coerce')
earthquakes['date_parsed'].sample(5)
earthquakes['date_parsed'].loc[earthquakes['date_parsed'].isnull()]
earthquakes['Date'].loc[[3378,7512,20650]]
earthquakes['date_parsed'][3378] = '1975-02-23'
earthquakes['date_parsed'][7512] = '1985-04-28'
earthquakes['date_parsed'][20650] = '2011-03-13'
earthquakes['date_parsed'].loc[earthquakes['date_parsed'].isnull()]
earthquakes.info()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head(10)
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes.head(10)
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
volcanos['Last Known Eruption'].shape
volcanos['date_parsed'] = volcanos[volcanos['Last Known Eruption'].str.endswith('CE')]['Last Known Eruption']
last_known_eruption_dates = volcanos['date_parsed'].dropna()
last_known_eruption_dates.sample(5)
eruption_dates_int = last_known_eruption_dates.apply(lambda x: -int(x[:-4]) if x.endswith('BCE') else int(x[:-3]))
eruption_dates_int #FINAL
sns.distplot(eruption_dates_int)

