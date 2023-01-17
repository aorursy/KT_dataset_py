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
earthquakes.info()
landslides.describe()
landslides.info()
volcanos.describe()
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
type(landslides['date'])
landslides['population'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
landslides['date_parsed'].dtype
# print the first few rows
landslides['date_parsed'].head()
landslides['date_parsed2']=pd.to_datetime(landslides['date'], infer_datetime_format=True)
landslides['date_parsed2'].head(10)
earthquakes['Date'].head()
earthquakes['date_parsed']=pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")
earthquakes['date_parsed'].head()
earthquakes.head()
#try to find that row-locate nearby
earthquakes[earthquakes['date_parsed']=='1975-02-23']
#check larger range-found it!
earthquakes.iloc[3370:3390]
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed']=pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_day_landslides = landslides['date_parsed'].dt.day
day_of_day_landslides.dtype
# Your turn! get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day

# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
dat_of_month_earthquakes = earthquakes['date_parsed'].dt.day
dat_of_month_earthquakes = dat_of_month_earthquakes.dropna()
sns.distplot(dat_of_month_earthquakes, kde=True, bins=31)
volcanos['Last Known Eruption'].sample(5)
volcanos['Last Known Eruption'].count()
volcanos['Last Known Eruption'][volcanos['Last Known Eruption']=='Unknown'].count()
volcanos['Last Known Eruption']=volcanos['Last Known Eruption'][volcanos['Last Known Eruption']!='Unknown']
volcanos.head(10)
volcanos_year=pd.DataFrame(volcanos['Last Known Eruption'].dropna())
volcanos_year.info()
volcanos_year.head()
volcanos_year['year'] = volcanos_year['Last Known Eruption'].str.split(' ', expand=True)[0] 
volcanos_year['mark'] = volcanos_year['Last Known Eruption'].str.split(' ', expand=True)[1]
volcanos_year.head()
volcanos_year.year.dtype
volcanos_year['year']=pd.to_numeric(volcanos_year['year'],errors='coerce').astype(np.int64)
volcanos_year.year.dtype
def change_mark(df):
    if df=='BCE':
        value=-1
    else:
        value=1
    return value
volcanos_year['mark2']=volcanos_year['mark'].apply(change_mark)
volcanos_year.head()
volcanos_year['year']=volcanos_year['year']*volcanos_year['mark2']
volcanos_year.head()
sns.distplot(volcanos_year['year'], kde=False, bins=10)
from scipy import stats
import matplotlib.pyplot as plt
# normalize the exponential data with boxcox
volcanos_year_normalized = stats.boxcox(volcanos_year['year']+15000)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(volcanos_year['year']+15000, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(volcanos_year_normalized[0], ax=ax[1])
ax[1].set_title("Normalized data")
