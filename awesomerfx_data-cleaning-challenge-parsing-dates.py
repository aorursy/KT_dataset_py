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
# take a look at some of the data
earthquakes.info()
landslides.info()
volcanos.info()
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
# print the first few rows of the date column
print(earthquakes['Date'].head())
# another method to check the data type of our Data column
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
print(earthquakes['date_parsed'].head())
earthquakes['date_parsed'].dtype
# try to get the day of the month from the date column
#day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
print (day_of_month_landslides.describe())
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
print (day_of_month_earthquakes.describe())
# remove na's
na_num = day_of_month_landslides.isnull().sum()
print(na_num)
if(na_num):
    day_of_month_landslides = day_of_month_landslides.dropna()
# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
# remove na's
na_num = day_of_month_earthquakes.isnull().sum()
print(na_num)
if(na_num):
    day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
volcanos['Last Known Eruption'].describe()
# data cleaning
# firstly, add a column called 'flag' aim to indicate the year (1 means CE;  0 means Unknown or missing; -1 means BCE)
volcanos['flag'] = volcanos['Last Known Eruption'].apply(lambda x: 0 if (x == 'Unknown') else -1 if(x.find('BCE') >= 0) else 1)
# Then, add a column called 'cleaned_date' aim to store the cleaned_year 
# change text'Unknown' to 0
volcanos['cleaned_date'] = volcanos['Last Known Eruption'].replace('Unknown', 0)
# split text'BCE' and 'CE'
volcanos['cleaned_date'] = volcanos['cleaned_date'].apply(lambda x: x.split('BCE')[0] if (str(x).find('BCE') >= 0) else x.split('CE')[0] if (str(x).find('CE') >= 0) else 0)

# take a look of Data together to compare
print(volcanos.loc[:, ('Last Known Eruption', 'cleaned_date', 'flag')].head())
volcanos['test'] = 1678
volcanos['parsed_date'] = pd.to_datetime(volcanos['test'], format='%Y')
print(volcanos['parsed_date'].head())