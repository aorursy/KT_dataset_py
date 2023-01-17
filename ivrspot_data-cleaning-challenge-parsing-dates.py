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
print('Earthquakes...')
print(earthquakes.head())
print(earthquakes['Date'].dtype)
# The type is also OBJECT
print('\nVolcanos...')
print(volcanos.head())
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['Date'].head()
print(earthquakes['Date'][10])
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'][10], format = "%m/%d/%Y")
earthquakes['date_parsed'].head()
#new dtype: datetime64[ns]
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
#day_of_month_landslides = landslides['date_parsed'].dt.day
#print(day_of_month_landslides.head())
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
month_of_landslides = landslides['date_parsed'].dt.month
print(month_of_landslides.head())
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
dmonth_of_earthquakes = earthquakes['date_parsed'].dt.day
print(dmonth_of_earthquakes.head())
# remove na's
dmonth_of_earthquakes = dmonth_of_earthquakes.dropna()

# plot the day of the month
sns.distplot(dmonth_of_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)
#Print samples of DB volcanos
#print('Volcanos sample...\n')
#print(volcanos.sample(5))
import matplotlib.pyplot as plt
#Clean data
# how many total missing values do we have?
missing_values_count = (volcanos == 'Unknown').sum()
total_cells = np.product(volcanos.shape)
total_missing = missing_values_count.sum()
# percent of data that is missing
print('\nPercent of data Unknown = '+str(float((total_missing/total_cells) * 100)) +' %')
#Information about data
# look at the # of missing points 
missing_values_count[0:]
#index unknown
index_unknown = volcanos['Last Known Eruption']=='Unknown'
vol_known = volcanos[index_unknown==False]['Last Known Eruption']
#change BCE and CE
#looks for values containing BCE    
BCE = [s for s in vol_known if "BCE" in s]
#removes BCE string
BCE = [x.strip(' BCE') for x in BCE]
#defines them as integers
BCE = list(map(int, BCE))
#add minus sign to BCE years
BCE = [ -x for x in BCE]
CE = [s for s in vol_known if " CE" in s]
CE = [x.strip(' CE') for x in CE]
CE = list(map(int, CE))
#merges the BCE and the CE integers to one list
mergedlist = BCE + CE
#plot the list
sns.distplot(mergedlist)
plt.xlabel("Year")
#Parsing Dates
#vol_known.head()
#volcanos['date_parsed'] = pd.to_datetime(vol_known, format = "%Y")