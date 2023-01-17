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
earthquakes['Date'].head() # dates are also of dtype: object

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'],  format = "%m/%d/%y")

# print the first few rows
#landslides['date_parsed'].head()
earthquakes.Time.dtype
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to double-check that the dtype is correct!)
#Note: Learned from 
#.apply(len)-- counts the length of string specified
earthquakes['lenDate'] = earthquakes['Date'].apply(len) #issue arises with dattime conversion, intuitivley you can first check whether the dates are properly formatted by counting lenght of string
earthquakes.loc[earthquakes['lenDate']>10]
#earthquakes.Date[earthquakes['Date'].str.contains('-')]

#replace corrupted data
earthquakes.loc[3378, 'Date']= '02/23/1975' #02:58:41
earthquakes.loc[7512, 'Date']= '04/28/1985' #02:53:41
earthquakes.loc[20650, 'Date']= '03/13/2011' #02:23:34

earthquakes.loc[3378, 'Time']= '02:58:41'
earthquakes.loc[7512, 'Time']= '02:53:41'
earthquakes.loc[20650, 'Time']= '02:23:34'
display(earthquakes.loc[[3378, 7512, 20650]])



earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")

#Combinign date and time colmuns and  parsing the resultant colum into a datetime object 
earthquakes['datetime']=earthquakes['Date']+' '+ earthquakes['Time']
earthquakes['datetime']=pd.to_datetime(earthquakes['datetime'], format= "%m/%d/%Y %H:%M:%S")
#earthquakes.sample(50)

# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
#day_of_month_landslides
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['datetime'].dt.day
day_of_month_earthquakes
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes .dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)

#volcanos['Last Known Eruption']