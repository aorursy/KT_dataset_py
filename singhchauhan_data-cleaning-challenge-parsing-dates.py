# modules we'll use
import pandas
import numpy
import seaborn
import datetime

# read in our data
earthquakes = pandas.read_csv("../input/earthquake-database/database.csv")
landslides = pandas.read_csv("../input/landslide-events/catalog.csv")
volcanos = pandas.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
numpy.random.seed(0)
# Taking a look at earthquakes data
earthquakes.info()
earthquakes.sample(10)
# Taking a look at landslides data
landslides.info()
landslides.sample(10)
# Taking a look at volcanos data
volcanos.info()
volcanos.sample(10)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
print(earthquakes['Date'].head())
# check the type of this column
earthquakes['Date'].dtype 
# that is also an 'object'
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pandas.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['Date'].dtype
earthquakes['Date'][0:10]
# Including new parsed column to the earthquakes data
earthquakes['Date_parsed'] = pandas.to_datetime(earthquakes['Date'])
earthquakes['Date_parsed'].head(10)
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column
day_earthquakes = earthquakes['Date_parsed'].dt.day
day_earthquakes.head()
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
seaborn.distplot(day_of_month_landslides, kde = False, bins = 31, color = "green")
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
# Dropping NA's
day_earthquakes = day_earthquakes.dropna()

# Visualizing it
seaborn.distplot(day_earthquakes, kde = False, color = "blue", bins = 31)
volcanos['Last Known Eruption'].sample(5)