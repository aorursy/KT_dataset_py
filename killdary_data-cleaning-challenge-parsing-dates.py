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
print(earthquakes['Date'].head())
print(earthquakes['Date'].dtype)
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%y")
# Following the tips of Steven Hu-https: //www.kaggle.com/n444466
# We searched for the rows where the date string was greater than 10 (00/00/0000)
earthquakes['lenDate'] = earthquakes['Date'].apply(len)
earthquakes.loc[earthquakes['lenDate'] > 10]
# Now knowing which lines are wrong, 
# We just need to correct the dates and take advantage of the correct time too
earthquakes.loc[3378, 'Date']= '02/23/1975' 
earthquakes.loc[7512, 'Date']= '04/28/1985'
earthquakes.loc[20650, 'Date']= '03/13/2011' 
earthquakes.loc[3378, 'Time']= '02:58:41' 
earthquakes.loc[7512, 'Time']= '02:53:41'
earthquakes.loc[20650, 'Time']= '02:23:34' 
earthquakes.loc[[3378, 7512, 20650]]
# Finally we can run the datetime format
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y") 
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
print(day_of_month_earthquakes.head())
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

sns.distplot(day_of_month_earthquakes, kde=False, bins=31)

volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")
volcanos['Last Known Eruption'].replace('Unknown','np.NaN', inplace = True)
volcanos.head(20)
newdf = volcanos['Last Known Eruption']
year = ["-"+x.strip(" BCE") if x.endswith("BCE") else x.strip(" CE") if x.endswith("CE") else np.NaN for x in newdf]
volcanos["Year"] = year
volcanos["Year"] = volcanos["Year"].astype(np.float64)
sns.distplot(volcanos["Year"].dropna(), kde = False)