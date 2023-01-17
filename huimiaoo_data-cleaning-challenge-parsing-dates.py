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
#identify inconsistent data.One way is through check the data length
earthquakes.rename(columns={'Date':'date','Time':'time'})
earthquakes['lenDate'] = earthquakes['Date'].apply(len)
inconsistent_index = earthquakes.loc[earthquakes['lenDate']>10]
#separate the two data set
earthquakes_selected = earthquakes.loc[earthquakes['lenDate'] < 11]
earthquakes_selected['date_parsed']= pd.to_datetime(earthquakes_selected['Date'],format ='%m/%d/%Y')
earthquakes_selected.sample(6)
#another way to check the data inconsistency is using pandas to infer the result

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
#step identify the inconsistent data

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format= False)
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
#insert bar chart
sns.barplot(day_of_month_earthquakes.values,day_of_month_earthquakes.index)
#distribution chart
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column

# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

volcanos.sample(5)
print(5*-1)
import re
def parse_year_volcanos(string_year):
    if string_year == 'Unknown':
        return 'NaN'
    else:
        year = int(re.search(r'\d+', string_year).group())
        if string_year[-3:] == 'BCE':
            year = year* -1
        return year
    
volcanos['parsed_last_known_eruption'] = volcanos['Last Known Eruption'].apply(parse_year_volcanos)
volcanos.sample()
def error():
    while True:
        volcanos['last_known_eruption_parsed']= pd.to_datetime(volcanos['Last Known Eruption'],format='%Y')
    else:
        return volcanos['Last Known Eruption'].values
error()