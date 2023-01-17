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
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True) # time data '1975-02-23T02:58:41.000Z' does not match format '%m/%d/%Y' (match)
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde = False, bins = 31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde = False, bins = 31)
volcanos['Last Known Eruption'].sample(5)
# Remove unknown and nan
volcanos['Last Known Eruption'] = volcanos['Last Known Eruption'].replace('Unknown', np.nan)
volcanos = volcanos.dropna()
volcanos['Last Known Eruption'].sample(5)
# select row of BCE and CE
BCE_row = volcanos['Last Known Eruption'].str.contains(' BCE')
CE_row = volcanos['Last Known Eruption'].str.contains(' CE')

# remove BCE and CE in string
BCE_series = volcanos['Last Known Eruption'][BCE_row].str.replace(' BCE', '')
CE_series = volcanos['Last Known Eruption'][CE_row].str.replace(' CE', '')
BCE_series = - BCE_series.astype(int)
CE_series = CE_series.astype(int)

series = BCE_series.append(CE_series)
volcanos['Last Known Eruption int'] = series
import seaborn as sns
sns.distplot(volcanos['Last Known Eruption int'])
def conv(x):
    return pd.Period(year = x, freq='A')
volcanos['Last Known Eruption period'] = volcanos['Last Known Eruption int'].apply(conv)
pd.PeriodIndex(volcanos['Last Known Eruption period'])
