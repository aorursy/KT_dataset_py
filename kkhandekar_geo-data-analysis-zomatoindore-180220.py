# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns

%matplotlib inline

import random

import re



# Geovisualization library

import folium

from folium.plugins import CirclePattern, HeatMap

data = pd.read_csv("../input/zomato-indore-data/zomato_indore.csv")
data.shape
#dropping the unnamed column

data = data.drop(['Unnamed: 0'],axis=1)
data.head()
data.describe()
#Checking for null/missing values

data.isna().sum()
#Checking for unique rating text

data.groupby('rating_text').size()
#Isolating the aggregate_rating & rating_text column

data_rating = pd.DataFrame(data, columns=['aggregate_rating','rating_text'])

#checking the newly created dataframe

data_rating.head()
#function to check the aggregate rating value for every type of rating

def rating_check():

  for x, y in data_rating.itertuples(index=False):

    std_rating = ['Average','Excellent', 'Good', 'Poor', 'Very Good']   #list of standard rating

    if any(y in stdr for stdr in std_rating):                           #checking if the value in rating column matches with that in the standard rating list

      print(y , ">>>", x)

    else:

      print(y, ">>>", "unknown rating") 



#calling the function

rating_check()
#function to update the rating as per the aggregate rating

def upd_rating():

  for x, y in data_rating.itertuples(index=False):

    std_rating = ['Average','Excellent', 'Good', 'Poor', 'Very Good']

    if any(y in stdr for stdr in std_rating):

      continue

    elif (0 <= x <= 2.4):

      data_rating['rating_text'] = data_rating['rating_text'].replace(y,std_rating[3])

    elif (2.5 <= x <= 3.4):

      data_rating['rating_text'] = data_rating['rating_text'].replace(y,std_rating[0])     

    elif (3.5 <= x <= 3.9):

      data_rating['rating_text'] = data_rating['rating_text'].replace(y,std_rating[2])     

    elif (4.0 <= x <= 4.4):

      data_rating['rating_text'] = data_rating['rating_text'].replace(y,std_rating[4])

    else:

      data_rating['rating_text'] = data_rating['rating_text'].replace(y,std_rating[1]) 



#calling the function

upd_rating()
#checking if the replace function has worked

data_rating.groupby('rating_text').size()
#combining the two dataframes and creating a new  

data_clean = data_rating.combine_first(data)
data_clean.head()
#getting the list of columns from original dataset = data

data.columns
#rearranging the columns as per the original

data_clean = data_clean[['name', 'locality', 'latitude', 'longitude', 'cuisines',

       'average_cost_for_two', 'aggregate_rating', 'votes', 'rating_text']]
data_clean.head()
#validating the rating_text column for the newly created dataframe

data_clean.groupby('rating_text').size()
#taking a backup

data_clean_backup = data_clean.copy()
#Cost vs Rating

sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(25,10))

ax = sns.swarmplot (x='average_cost_for_two', y='aggregate_rating', data=data_clean, hue = 'rating_text')

plt.title('Average Costs for Two v/s Aggregate Rating')

plt.ylabel('Aggregate Rating')

plt.xlabel('Average Costs for Two')
sns.set(style="darkgrid")

fig = plt.figure()

fig = sns.relplot(x="aggregate_rating", y="votes", kind="line", hue="rating_text",data=data_clean,height=15,aspect=1,palette="cubehelix")

fig.fig.set_size_inches(15,10)

fig.set_titles("Votes Distribution Per Rating")

fig.set_xlabels("Aggregate Rating")

fig.set_ylabels("Total Votes")

plt.show()

#Function to generate base map

def BaseMap(default_location=[22.719568, 75.857727], default_zoom_start=12):   #the default location co-ordinates are of Indore

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start, width='50%', height='50%')

    return base_map
base_map = BaseMap()

base_map
#creating a dataframe with condition#1

cond1 = (data_clean['average_cost_for_two'] <= 500) & (data_clean['aggregate_rating'] >= 4.0)

df_cond1 = data_clean[cond1]

df_cond1['popup_text'] = df_cond1['name'] + "," + "₹"+df_cond1['average_cost_for_two'].astype(str) + "," + df_cond1['rating_text']

df_cond1.head()
#creating a dataframe with condition#2

cond2 = (data_clean['average_cost_for_two'] >= 501) & (data_clean['average_cost_for_two'] <= 1500) & (data_clean['aggregate_rating'] >= 4.0)

df_cond2 = data_clean[cond2]

df_cond2['popup_text'] = df_cond2['name'] + "," + "₹"+df_cond2['average_cost_for_two'].astype(str) + "," + df_cond2['rating_text']

df_cond2.head()
#creating a dataframe with condition#3

cond3 = (data_clean['average_cost_for_two'] >= 1501) & (data_clean['average_cost_for_two'] <= 3000) & (data_clean['aggregate_rating'] >= 4.0)

df_cond3 = data_clean[cond3]

df_cond3['popup_text'] = df_cond3['name'] + "," + "₹"+df_cond3['average_cost_for_two'].astype(str) + "," + df_cond3['rating_text']

df_cond3.head()
#creating a dataframe with condition#4

cond4 = (data_clean['average_cost_for_two'] >= 500) & (data_clean['average_cost_for_two'] <= 1000) & (data_clean['rating_text'].isin(['Average','Poor']))

df_cond4 = data_clean[cond4]

df_cond4['popup_text'] = df_cond4['name'] + "," + "₹"+df_cond4['average_cost_for_two'].astype(str) + "," + df_cond4['rating_text']

df_cond4.head()


df_cond1.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]], popup=row["popup_text"]).add_to(base_map), axis=1)

base_map
df_cond2.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]], popup=row["popup_text"]).add_to(base_map), axis=1)

base_map
df_cond3.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]], popup=row["popup_text"]).add_to(base_map), axis=1)

base_map
df_cond4.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]], popup=row["popup_text"]).add_to(base_map), axis=1)

base_map