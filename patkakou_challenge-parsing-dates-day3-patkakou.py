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
# look on the the earthquakes dataframe
earthquakes.head()
earthquakes.info()
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].head()
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
# earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y") ==> returns ValueError: time data '1975-02-23T02:58:41.000Z' does not match format '%m/%d/%Y' (match)
# Finding where the date in some fields is poorly formated
earthquakes.Date[earthquakes['Date'].str.contains('-')] # 3 rows contain '-'
# Let fix it by replacing with correct date format
earthquakes.Date = earthquakes.Date.replace('1975-02-23T02:58:41.000Z', '02/23/1975')
earthquakes.Date = earthquakes.Date.replace('1985-04-28T02:53:41.530Z', '04/28/1985')
earthquakes.Date = earthquakes.Date.replace('2011-03-13T02:23:34.520Z', '04/13/2011')
#Let check again 
earthquakes.Date[earthquakes['Date'].str.contains('-')] # Great 
# Now let create a new column, date_parsed, in the earthquakes
earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], format = '%m/%d/%Y')
earthquakes.Date_parsed.head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
Day_of_Month_earthquakes = earthquakes["Date_parsed"].dt.day
Day_of_Month_earthquakes.sample(5)
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
Day_of_Month_earthquakes = Day_of_Month_earthquakes.dropna()

#Plotting the days of the month

sns.distplot(Day_of_Month_earthquakes, kde=False, bins=31)

volcanos.head()
volcanos['Last Known Eruption'].value_counts()
# 637 'Last Known Eruption' are unknown and that correspond to almost 42% (637/1508) of data in this column
# it will not be a good idea to delete them but how to figure out
Unknown_Eruption = volcanos.loc[volcanos['Last Known Eruption'] == 'Unknown']
Unknown_Eruption.head()
Unknown_Eruption['Name'].value_counts()
