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
volcanos.info()

volcanos.head(5)
earthquakes.info()
earthquakes.head(5)
# create a new column, date_parsed, with the parsed dates
# date is in the format 12/31/72 (dd/mm/yy)
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
landslides.info()
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# earthquakes // 01/02/1965
# date is in the format 12/31/72 (dd/mm/yyyy)

# note that the following didn't work because there are at least two different types of dates:
# 01/02/1965 and 1975-02-23T02:58:41.000Z
# earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")
#
# so, I tried the suggested 'infer_datetime_format' option as recomended in the text above.
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)

# all good, double check with:
earthquakes.info()




# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes.head()
# remove na's
# we need to drop the na's because otherwise we'll get an error with the plot below.
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

# remove na's
# removing na's so that we can plot below without errors.
# so, instead, I did it inline within the plot so that it was more obvious... 
# day_of_month_earthquakes = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides.dropna(), kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)