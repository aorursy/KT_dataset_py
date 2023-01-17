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
print(earthquakes.columns)
print(earthquakes['Date'].head())
print('Earthquake Date column type {0}'.format(earthquakes['Date'].dtype))


# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)
earthquakes['date_parsed'].head

# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes.head
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
last_known_eruption = volcanos['Last Known Eruption']
print(last_known_eruption.shape)
last_known_eruption_dropped = last_known_eruption.replace('Unknown', np.nan).dropna()
print('Shape {0}'.format(last_known_eruption_dropped.shape))
print('Data {0}'.format(last_known_eruption_dropped.head))
def computeDate(s):
    #looks for values containing BCE
    if "BCE" in s:
        #removes BCE string        
        #defines them as integers
        return -int(s.strip(' BCE'))
    if " CE" in s:        
        return int(s.strip(' CE'))        
last_known_eruption_cleanDate = last_known_eruption_dropped.apply(computeDate)
#plot the list
sns.distplot(last_known_eruption_cleanDate)
