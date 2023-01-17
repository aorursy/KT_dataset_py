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

# print the first few rows of the date column
print(landslides['date'].head())

# check the data type of our date column
landslides['date'].dtype

earthquakes.columns

# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype




# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
landslides['date'].head()
earthquakes['Date'].sample(10)
# print the first few rows
#landslides['date_parsed'].head()
earthquakes['Date'].head()
len('1975-02-23T02:58:41.000Z')
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
#earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")
#earthquakes = earthquakes[earthquakes['Date'] != '1975-02-23T02:58:41.000Z']
earthquakes['Date_len'] = earthquakes['Date'].apply(lambda x : len(x))
earthquakes = earthquakes[earthquakes['Date_len']!=24]
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")



# try to get the day of the month from the date column
day_of_month_landslides = landslides['date_parsed'].dt.day


# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
len(day_of_month_earthquakes.dropna()) == len(day_of_month_earthquakes)




# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)


# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
#Check if a digit in present in the string if it is then split the string by a single space and return the year else return unknow
volcanos['Last Known Eruption year'] = volcanos['Last Known Eruption'].apply(lambda x : x.split()[0] if any([i.isdigit() for i in x]) else 'Unknown')

volcanos['Last Known Eruption year']



















