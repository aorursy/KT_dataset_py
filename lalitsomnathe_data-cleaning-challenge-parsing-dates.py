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
from IPython.display import display
display(earthquakes.head(2))
display(volcanos.head(2))
print(volcanos.columns) # NO date type column
landslides.sample(2)
# print the first few rows of the date column
print(landslides['date'].head())
landslides.dtypes
landslides.info()
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes.sample(2)
earthquakes['Date'].dtypes
landslides.sample(2)
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
#landslides['date_parsed'].dtype
earthquakes.sample(2)
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

#earthquakes['date_parsed']= pd.to_datetime(earthquakes['Date'],format = "%m/%d/%Y" )
earthquakes['date_parsed']=pd.to_datetime( earthquakes['Date'] )

earthquakes['date_parsed'].head()
display(landslides.head(2))
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date_parsed'].dt.day
display(day_of_month_landslides.head(2))
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
display(earthquakes.head(2))
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
display(day_of_month_earthquakes.head(2))
day_of_month_landslides.isnull().sum()
day_of_month_landslides.loc[day_of_month_landslides.isnull()]
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()
import matplotlib.pyplot as plt
# plot the day of the month
fig,ax=plt.subplots(1,2)

sns.distplot(day_of_month_landslides,  kde=False,bins=31 ,ax=ax[0])
sns.distplot(day_of_month_landslides ,ax=ax[1])
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
#day_of_month_earthquakes.isnull().sum()
day_of_month_earthquakes = day_of_month_earthquakes.dropna()
import matplotlib.pyplot as plt
# plot the day of the month
fig,ax=plt.subplots(1,2)

sns.distplot(day_of_month_earthquakes,  kde=False,bins=31 ,ax=ax[0])
sns.distplot(day_of_month_earthquakes ,ax=ax[1])
#earthquakes['date_parsed'].dt.month

volcanos['Last Known Eruption'].sample(5)
