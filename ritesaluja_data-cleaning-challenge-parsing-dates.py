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
print(earthquakes['Date'])
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parseda'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y", errors = 'coerce')
earthquakes['date_parseda'].head()
#for i in earthquakes['Date']:
#    if i == '1975-02-23T02:58:41.000Z':
 #       print(i)
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parseda'].dt.day
print(day_of_month_earthquakes)
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
volcanos['Last Known Eruption'].sample(5)
volcanos['date_parsed'] = volcanos[volcanos['Last Known Eruption'].str.endswith('CE')]['Last Known Eruption']
last_known_eruption_dates = volcanos['date_parsed'].dropna()
eruption_date_lowest = 1
eruption_dates_bce_int = last_known_eruption_dates.apply(lambda x: int(x[:-4]) if x.endswith('BCE') else int(0))
for i in  eruption_dates_bce_int:   
    if i > eruption_date_lowest:
        eruption_date_lowest = i

print(eruption_date_lowest)
print(last_known_eruption_dates)              
    
pd.to_datetime(last_known_eruption_dates, format='%y',unit='D',
                   origin='julian')
