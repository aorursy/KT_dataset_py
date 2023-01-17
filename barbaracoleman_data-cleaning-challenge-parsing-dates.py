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
earthquakes.head()
print(earthquakes['Date'].head(20))
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
earthquakes['lenDate'] = earthquakes['Date'].apply(len)
earthquakes.loc[earthquakes['lenDate'] > 10]
earthquakes_with_normal_dates = earthquakes.loc[earthquakes['lenDate'] < 11]
earthquakes_with_long_dates = earthquakes.loc[earthquakes['lenDate'] > 10]
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
#earthquakes['date_parsed'] = pd.to_datetime(earthquakes_with_normal_dates['Date'], format = "%m/%d/%Y")
#earthquakes['date_parsed'] = pd.to_datetime(earthquakes_with_long_dates['Date'], format = "%Y-%m-%dT%H:%M:%S.%f")

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
earthquakes.head()
earthquakes.loc[earthquakes['lenDate'] > 10]
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
#day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes[0:10]
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
# set unknown values to NaN
volcanos['Last Known Eruption'] = volcanos['Last Known Eruption'].replace('Unknown', np.NaN)
# set new column LKEyear to the first four digits in Last Known Eruption as an float
volcanos['LKEyear'] = pd.to_numeric(volcanos['Last Known Eruption'].str[0:4], errors='coerce')

# then multiply the value by -1 if it is BCE, otherwise leave as positive
volcanos['LKEyear'] = volcanos.apply(lambda x: -1*x['LKEyear'] if 'BCE' in str(x['Last Known Eruption']) else x['LKEyear'], axis=1)
volcanos['LKEyear'].head()

volcanos.head()