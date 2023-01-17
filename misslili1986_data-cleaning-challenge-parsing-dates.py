# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
#landslides.sample(6)
#earthquakes.head()
# set seed for reproducibility

np.random.seed(0)
# print the first few rows of the date column
print(landslides['date'].head())

#print(landslides['population'].head())
# check the data type of our date column
landslides['date'].dtype
#landslides['population'].dtype
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
earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)
#earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%y")
earthquakes['Date_parsed'].head()
# try to get the day of the month from the date column
#day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column of earthquakes data
day_of_month_earthquakes = earthquakes['Date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos.head()
volcanos.sample(5)

volcanos['Last Known Eruption'].head()
#volcanos['Last Know Eruption_parsed'] = pd.to_datetime(volcanos['Last Know Erupation'], infer_datetime_format = True)
eruption_year = volcanos['Last Known Eruption']
eruption_year_unknowntoNone = eruption_year.replace({'Unknown':None})
eruption_year_unknowntoNone.isnull().sum()
# build a new data frame to extract Last known eruption info
eruption_year_n = eruption_year_unknowntoNone.str.split(' ',expand=True)
eruption_year_n.columns = ['year', 'era']
eruption_year_n.head()

# convert column year to numerical datatype
#check the datatype of the original data
eruption_year_n['year'].dtype
#convert the datatype then check the dtype
eruption_year_n['year'] = pd.to_numeric(eruption_year_n.year)
eruption_year_n['year'].dtype

# work on the era column:set BCE as -1 while CE as 1
eruption_year_n['era'] = eruption_year_n['era'].replace({'BCE':-1, 'CE':1})
eruption_year_n['era'].head()
#multiply year with era column to get the final eruption year data
real_eruption_year = eruption_year_n.year * eruption_year_n.era
real_eruption_year.head()

# plot the eruption year
# first we need to drop the NaN entry
sns.distplot(real_eruption_year.dropna(), kde = False)