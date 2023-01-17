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
# for earquake dataset
earthquakes.info()
earthquakes.head()

# for landslides dataset
landslides.info()
landslides.head()

# for volcanos dataset
volcanos.info()
volcanos.head()
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

#earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y") -> returns a error date in some fields in poorly formated

# Find where the date is poorly formated
earthquakes.Date[earthquakes['Date'].str.contains('-')] # its poorly formated in 3 fields only

#Replacing the field with date in different format ( i dont know a better way of doing it) -> Improve here
earthquakes = earthquakes.replace('1975-02-23T02:58:41.000Z', '02/23/1975')
earthquakes = earthquakes.replace('1985-04-28T02:53:41.530Z', '04/28/1985')
earthquakes = earthquakes.replace('2011-03-13T02:23:34.520Z', '03/13/2011')
earthquakes.Date[earthquakes['Date'].str.contains('-')]

# Finaly do the task
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y") # Date format is diff from landslides DataFrame
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes
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
volcanos['Last Known Eruption'].sample(5)
# But in how many 'Last Known Eruption' == 'Unknown'
volcanos.info() # no help from this because 'Unknown' != nan
isunknown = volcanos['Last Known Eruption'] == 'Unknown'
isunknown.sum() # ah! ok in 637 the 'Last Known Eruption' == 'Unknown'
# Why are some 'Last Known Eruption' == 'Unknown'? 
# Most likely because the date is really unknown and not due to lack of anotation
# So can we intrapolate the date?
volcanos['Last Known Eruption']
# The data is not in cornological order so 'bfill' or 'ffill' would be useless. So probably not. 
# Lets just treat it as nan. Lets replace 'Unknown' by nan
volcanos = volcanos.replace('Unknown', np.nan)
volcanos.info() # nice now 'Unknown' is treated as missing value
unkown_out_volcanos = volcanos.dropna()
unkown_out_volcanos.info() # now we only have the rows with known date
# Separating the old data to extract and transform the year
processing_dates = pd.DataFrame() # need a better way to do this
processing_dates['LastEruptionYear'] = unkown_out_volcanos['Last Known Eruption'].str.split(' ', expand=True)[0] # just year
processing_dates['LastEruptionPeriod'] = unkown_out_volcanos['Last Known Eruption'].str.split(' ', expand=True)[1] # just the period
processing_dates.head() # looks good
volcanos.columns
# I think that would be nice if we see these two columns in the original dataframe
volcanos = pd.concat([volcanos,processing_dates],axis=1)
volcanos
volcanos=volcanos.dropna()
# How many 'options' tehre is for period?
processing_dates['LastEruptionPeriod'].unique() # 2, makes sence

BCE = list(processing_dates.LastEruptionYear[processing_dates['LastEruptionPeriod'].str.match('BCE')]) # just those BCE
CE = list(processing_dates.LastEruptionYear[processing_dates['LastEruptionPeriod'].str.match('CE')]) # just those BC
CE = [int(x) for x in CE ] # i got a strange error from this list items format so i did this

new_BCE = [] # convert BCE to negative values, so before current time
for date in BCE:
    new_BCE.append(0-int(date))
new_period = np.array(new_BCE + CE)
sns.distplot(new_period, kde=True, bins=31) # and done
