# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanoes = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)
earthquakes.head()
landslides.head()
volcanoes.head()
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
earthquakes['date_parsed']=pd.to_datetime(earthquakes['Date'],infer_datetime_format=True)
#earthquakes['Date'].dtype
earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes=earthquakes['date_parsed'].dt.day
day_of_month_earthquakes
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes=day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes,kde=False,bins=31)
volcanoes['Last Known Eruption'].sample(5)
import re

index_unknown=[]

for i,j in enumerate(volcanoes['Last Known Eruption']):
    if re.match(j,'Unknown'):
        index_unknown.append(i)
        
#index_unknown
volcanoes=volcanoes.drop(volcanoes.index[index_unknown]).reset_index(drop=True)
#volcanoes.reset_index(drop=True)
before_christ=[]
after_christ=[]
for i in volcanoes['Last Known Eruption']:
    lst=i.split()
    if lst[1]=='BCE':
        before_christ.append(lst[0])
    else:
        after_christ.append(lst[0])
import matplotlib.pyplot as plt

before_christ=pd.Series(before_christ)
#fig,ax=plt.subplots(1,2)
#sns.distplot(before_christ,ax=ax[0])
#ax[0].set_title('Before Christ')
#sns.distplot(after_christ,ax=ax[1])
#ax[1].set_title('After Christ')
plt.figure(figsize=(80,30))
before_christ.value_counts().plot(kind='bar')

after_christ=pd.Series(after_christ)
plt.figure(figsize=(80,30))
after_christ.value_counts().plot(kind='bar')
