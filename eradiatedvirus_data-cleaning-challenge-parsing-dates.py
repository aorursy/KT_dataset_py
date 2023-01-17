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
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# there are three dates that don't match the pattern mm/dd/yyyy. The code below finds these records.
earthquakes['lenDate'] = earthquakes['Date'].apply(len)

earthquakes.loc[earthquakes['lenDate'] > 10]

# create a data frame with the three records with nonconfonforming dates removed
earthquakes_with_normal_dates = earthquakes.loc[earthquakes['lenDate'] < 11]
#Code to pull the three other values and parse them. I decided to just eliminate the rows entirely. 

#earthquakes_with_normal_dates = earthquakes.loc[earthquakes['lenDate'] > 10]
#pd.to_datetime(earthquakes_with_normal_dates['Date'], format = "%Y/%m/%d")
earthquakes_with_normal_dates['Date_parsed'] = pd.to_datetime(earthquakes_with_normal_dates['Date'], format = "%m/%d/%Y")
earthquakes_with_normal_dates['Date_parsed'].sample(5)
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes_with_normal_dates['Date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

sns.distplot(day_of_month_earthquakes, bins = 31, kde=False)
volcanos['Last Known Eruption'].sample(5)
import re
import numpy as np
def parse_year_volcanos(string_year):
    if string_year == "Unknown":
        return np.NaN
    else:
        year =  int(re.search(r'\d+', string_year).group())
        if string_year[-3:] == 'BCE':
            year = year * -1
        return year

volcanos['Parsed_year']  =  volcanos['Last Known Eruption'].apply(parse_year_volcanos)
volcanos.hist(column='Parsed_year', grid=False)