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
earthquakes.sample(3)
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

#earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")
#Providing the format does not work as there are some values like ''1975-02-23T02:58:41.000Z'
#Better leave it to pandas to determine the format

earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'])
#print (earthquakes['Date_parsed'].dtype)
earthquakes['Date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
#landslides['date_parsed']
# Your turn! get the day of the month from the date_parsed column

#earthquakes['Date_parsed']
day_of_month_earthquakes = earthquakes['Date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

#day_of_month_earthquakes.isnull().sum()
sns.distplot(day_of_month_earthquakes, bins=31, kde=False)
#volcanos['Last Known Eruption'].sample(5)
volcanos.sample(5)
#Let me see how the categorical values are distributed...atleast for 'Last Known Eruption' column
volcanos['Last Known Eruption'].describe(include=['O'])

#Seems like unknown is top used with 637 times out of 1508 with (637/1508) or 42% usage
#print (637/1508)

#Let me decide to replace all unknows with 1900 as I can take it as starting of 19th century
#First select all the dates where it is 'unknown'
volcanos_copy = volcanos.copy()
unknown_dates = volcanos_copy[volcanos_copy['Last Known Eruption'] == 'Unknown']

#volcanos_copy['Last Known Eruption'][volcanos_copy['Last Known Eruption'] == 'Unknown'].size
volcanos_copy['Last Known Eruption'][volcanos_copy['Last Known Eruption'] == 'Unknown'] = 1900
#Check the size of unknown, it 0 now
for index, row in volcanos_copy.iterrows():
    #print (index, type (row))
    if isinstance(row['Last Known Eruption'], str) and row['Last Known Eruption'].endswith(' CE'): 
        #volcanos_copy['Last Known Eruption'].replace(row['Last Known Eruption'], row['Last Known Eruption'][:-3], inplace=True)
        volcanos_copy['Last Known Eruption'][volcanos_copy['Last Known Eruption'] == row['Last Known Eruption']] = row['Last Known Eruption'][:-3]
