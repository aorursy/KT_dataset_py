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
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format='%m/%d/%Y',errors='coerce')
earthquakes['date_parsed']
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides
# Your turn! get the day of the month from the date_parsed column
earthquakes['date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_the_month_earthq = earthquakes['date_parsed'].dt.day.dropna()

ax= sns.distplot(day_of_the_month_earthq, bins=15)
volcanos['Last Known Eruption'].sample(5)
volcanos['Last Known Eruption'] = volcanos['Last Known Eruption'].replace('Unknown',np.nan)
'Unknown'  in volcanos['Last Known Eruption']
volcanos.dropna(subset=['Last Known Eruption'],inplace=True)
volcanos['Last Known Eruption'].sample(10)
col = volcanos['Last Known Eruption']
NewCol = list()
NewDates = list()
for item in col:
    if ' BCE' in item:
        item = item.replace(' BCE', '')
        items = 0-int(item)
        NewCol.append(items)
    elif ' CE' in item:
        item = item.replace(' CE', '')
        itemz = int(item)
        NewCol.append(itemz)
    else:
        NewCol.append(item)
        
NewCol
ax =sns.distplot(NewCol)
ax.set_title('Last Known Eruption Dist')
ax.set_xlabel('TimeLineInYears')
NewCol
volcanos['Last Known Eruption NewCol']= NewCol
volcanos

