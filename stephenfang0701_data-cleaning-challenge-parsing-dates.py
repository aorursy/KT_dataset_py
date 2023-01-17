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
landslides.columns
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
earthquakes.columns
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].head()
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
earthquakes['Date'].head()
earthquakes[earthquakes['Date'] == '1975-02-23T02:58:41.000Z']
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# row 3378 exists an irregular datetime format, thus to_datetime function needs to input the infer_datetime_format argument
earthquakes['Datetime'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
print(earthquakes[['Datetime', 'Date']].head())
print(earthquakes[['Datetime', 'Date']].dtypes)
# select the row with irregular date format
irregular_row = earthquakes[~earthquakes['Date'].str.contains('[0-9]{2}/[0-9]{2}/[0-9]{4}')]
irregular_row.head()
earthquakes.loc[irregular_row.index][['Datetime', 'Date']]
reg_earthquakes = earthquakes.drop(irregular_row.index)
# to_datetime function tries to parse the string with the format specified first, then use infer_datetime_format if failed
# reg_earthquakes dataframe doesn't contain any irregular date format
%timeit pd.to_datetime(reg_earthquakes['Date'], format='%m/%d/%Y')
%timeit pd.to_datetime(reg_earthquakes['Date'], format="%m/%d/%Y", infer_datetime_format=True)

# parsing date with infer_datetime_format runs a bit slower than specifying the string format
%timeit pd.to_datetime(reg_earthquakes['Date'], infer_datetime_format=True)
# Parsing the date with different format
%timeit pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y", infer_datetime_format=True)
%timeit pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
# try to get the day of the month from the date column
# landslides is not datetime format!
# day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['Datetime'].dt.day
day_of_month_earthquakes.head()
reg_earthquakes['Datetime'] = pd.to_datetime(reg_earthquakes['Date'], format='%m/%d/%Y')
reg_earthquakes['Datetime'].dt.day.head()
day_of_month_landslides[day_of_month_landslides.isnull()]
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
earthquakes[earthquakes['Datetime'].isnull()]
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
def to_year(s):
    if s == "Unknown":
        return 0
    else:
        year, unit = s.split()
        return -int(year) if unit == "BCE" else int(year)
volcanos['parsed Last Known Eruption'] = volcanos['Last Known Eruption'].apply(to_year)
volcanos['parsed Last Known Eruption'].head()

parsed_year = volcanos['parsed Last Known Eruption']
sns.distplot(parsed_year.drop(parsed_year[parsed_year==0].index), kde=False)
replace_bce = lambda s: '-'+s.group(0).split()[0]
replace_ce = lambda s: s.group(0).split()[0]
replace_year = volcanos_year['Last Known Eruption']
replace_year = replace_year.str.replace('[0-9]+ BCE', replace_bce)
replace_year = replace_year.str.replace('[0-9]+ CE', replace_ce) 
replace_year.head()