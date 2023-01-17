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
earthquakes.Date.dtype, earthquakes.Date.head()
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
date_hour_index = []
date_index = []
for i, date in enumerate(earthquakes.Date):
    if len(date)==10: date_index.append(i)
    elif len(date)==24: date_hour_index.append(i)
        
len('02/23/1975'), len('1975-02-23T02:58:41.000Z'), '1975-02-23T02:58:41.000Z'[:10]

earthquakes.loc[date_hour_index,'date_parsed'] = pd.to_datetime(earthquakes.loc[date_hour_index,'Date'], format= '%Y-%m-%dT%I:%M:%S.%fZ')
earthquakes.loc[date_index,'date_parsed'] = pd.to_datetime(earthquakes.loc[date_index,'Date'], format= '%m/%d/%Y')
earthquakes['date_parsed'].isnull().sum()
earthquakes.loc[earthquakes.date_parsed==None,'date_parsed']
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['date_parsed'].dt.date)
earthquakes.date_parsed.head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes.head()
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes.dropna(inplace=True)

sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
unknown_index = volcanos[volcanos['Last Known Eruption'] == 'Unknown'].index
volcanos.loc[unknown_index, 'Last Known Eruption'] = np.nan
volcanos['Last Known Eruption Year'] = volcanos['Last Known Eruption'].apply(lambda x: str(x).split(' ')[0] if not pd.isnull(x) else np.nan)
volcanos['Last Known Eruption BCE/CE'] = volcanos['Last Known Eruption'].apply(lambda x: str(x).split(' ')[1]  if not pd.isnull(x) else np.nan)
volcanos
bce_index = volcanos[volcanos['Last Known Eruption BCE/CE']=='BCE'].index
volcanos.loc[bce_index, 'Last Known Eruption Year'] = volcanos.loc[bce_index, 'Last Known Eruption Year'].astype(int) + 2018
ce_index = volcanos[volcanos['Last Known Eruption BCE/CE'] == 'CE'].index
volcanos.loc[ce_index, 'Last Known Eruption Year'] = volcanos.loc[ce_index, 'Last Known Eruption Year'].astype(int)
volcanos
