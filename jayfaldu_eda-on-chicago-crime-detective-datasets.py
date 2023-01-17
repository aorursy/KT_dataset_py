import time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# change destination according to your data



df = pd.read_csv('/kaggle/input/chicago-crime-detective/Chicago_Crime_Detective.csv',parse_dates=['Date'],infer_datetime_format=True)

df.head()

df.Date[0]      # converted in time sucessfully
df.Date[0].to_pydatetime().month
df.dtypes
month=[]

for i in range(len(df.Date)):

    month.append(df.Date[i].to_pydatetime().month)
df.insert(10,'Month',month)
df
df.describe(include=['object'])
df['Unnamed: 0'].median()
df[df['Unnamed: 0'] == df['Unnamed: 0'].median()]
df
df_month=df.groupby(['Month'])
df_month.count().ID
df_month.ID.count().plot(kind='bar',figsize=(5,7))

plt.xlabel('month')

plt.ylabel('number of crime made')
df_month_as_index =df.set_index(['Month'])  



# set month as index (its done for count frequancy of month) it's 2nd method to give ansof the above question
df_month_as_index.index.value_counts()  



# show frequancy of month, we can see less frequent month as february
df_month_as_index.index.value_counts().plot(kind='bar',figsize=(5,7))

plt.xlabel('month')

plt.ylabel('number of crime made')
df
df.Date[0].to_pydatetime().weekday()  # return weekday in timestamp
# make a list of weekday of all entry in df and add as column in data frame



weekday = []

for i in range(len(df.Date)):

    weekday.append(df.Date[i].to_pydatetime().weekday())



df.insert(11,'weekday',weekday)



df.head()
weekday_as_index = df.set_index(['weekday'])

weekday_as_index.head()
weekday_as_index.index.value_counts()
weekday_as_index.index.value_counts().plot(kind='bar',figsize=(5,7))

plt.xlabel('weekdays:- \n 0-monday & 6-sunday')

plt.ylabel('number of crime made')
df_arrestmade = df_month_as_index[df_month_as_index.Arrest==True]

df_arrestmade.head()
df_arrestmade.index.value_counts()
year_as_index = df.set_index('Year')

year_as_index.head()
year_count=year_as_index.groupby('Year').count()

year_count
year_count.ID.plot(kind='bar',figsize=(10,7))

plt.xlabel('year')

plt.ylabel('number of crime made')
year_wise_arrestmade = year_as_index[year_as_index.Arrest==True]
year_wise_arrestmade[year_wise_arrestmade.index<2007].ID.count()  ## 2001 to 2006
year_wise_arrestmade[year_wise_arrestmade.index>2006].ID.count()  ## 2007 to 2012
year_as_index.head()
groupby_year = year_as_index.groupby('Year')
groupby_year.head()
total_crime_by_year = groupby_year.count()

total_crime_by_year
groupby_year_arrestmade = year_as_index[year_as_index.Arrest==True].groupby('Year')
total_arrest_by_year = groupby_year_arrestmade.count()

total_arrest_by_year
total_arrest_by_year.ID.plot(kind='bar',figsize=(5,7))

plt.xlabel('year')

plt.ylabel('number of arrest made')
arrest_proportion = total_arrest_by_year.ID/total_crime_by_year.ID

arrest_proportion
arrest_proportion.plot(kind='bar',figsize=(5,7))

plt.xlabel('year')

plt.ylabel('proportion of arrest made against crime happned')
location_as_index = df.set_index('LocationDescription')

location_as_index.head()
groupby_location = location_as_index.groupby('LocationDescription')

groupby_location.head()
count_by_location = groupby_location.ID.count().sort_values(ascending=False)

count_by_location
count_by_location.head(25)
top5_location = ['STREET','PARKING LOT/GARAGE(NON.RESID.)','ALLEY','GAS STATION','DRIVEWAY - RESIDENTIAL']



Top5 = df[df.LocationDescription.isin(top5_location)]
Top5
weekday_as_index.head()
weekday_as_index[weekday_as_index.LocationDescription=='GAS STATION'].index.value_counts()