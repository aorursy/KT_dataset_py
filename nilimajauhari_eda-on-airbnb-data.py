# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing required libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

import plotly.express as px

import plotly.graph_objects as go

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor
# Reading the data set

airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# Let us take a look at the dimensions of the data

airbnb.shape
# Visualizing the top 5 rows of the data set

airbnb.head(5)
# Check the data types of each attribute

airbnb.dtypes
# Removing the attributes which are not required for the analysis

airbnb.drop(['id','host_name','last_review'], axis = 1, inplace = True)
# Let us now check for missing values in the data set

airbnb.isnull().sum()
# Replacing missing values in the reviews_per_month column

airbnb.fillna({'reviews_per_month':0}, inplace = True)



# Replacing missing values in the name column

airbnb.fillna({'name':'Not Mentioned'}, inplace = True)
# Let us again check whether all the missing values have been removed or not

airbnb.isnull().sum()
# Plotting the correlation matrix

plt.figure(figsize = (15,15))

sns.heatmap(airbnb.corr(), annot=True)

plt.show()
# Let us now take a look at the different neighborhood groups

airbnb['neighbourhood_group'].unique()
# Let us now take a look at how many properties are there in each neighbourhood group

sns.countplot(x = 'neighbourhood_group', data = airbnb) 
# Let us now take a look at different room types being offered at the properties

airbnb['room_type'].unique()
# Let us now look at what type of rooms do most properties offer

sns.countplot(x = 'room_type', data = airbnb) 
# Let us now take a look at the price range of the properties

print("--- Price Per Night ---")

print("Minimum Price in $:", min(airbnb['price']))

print("Maximum Price in $:", max(airbnb['price']))

print("Average Price in $:", airbnb['price'].mean())
# Let us take a look at the minimum number of nights required to be booked at the properties

airbnb['minimum_nights'].unique()
# No of listings a particular host has

print("Minimum number of listings a particular host has:", airbnb['calculated_host_listings_count'].min())

print("Maximum number of listings a particular host has:", airbnb['calculated_host_listings_count'].max())
# Let us take a look at the number of listings a host has on Airbnb

airbnb['calculated_host_listings_count'].unique()
# Visualizing the percentage of listings most host have

labels = airbnb['calculated_host_listings_count'].unique()

sizes = airbnb['calculated_host_listings_count'].value_counts()*100



plt.figure(figsize = (30,30))

fig, ax = plt.subplots()

ax.pie(sizes, labels = labels, autopct = '%1.1f%%')

ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

ax.set_title('Listings a host has on Airbnb')





plt.show()
fig = px.scatter_mapbox(airbnb, lat = "latitude", lon = "longitude", hover_name = "neighbourhood", hover_data = ["neighbourhood_group", "price"],

                        color_discrete_sequence = ["fuchsia"], zoom = 3, height = 300)

fig.update_layout(mapbox_style = "open-street-map")

fig.update_layout(margin = {"r":0,"t":0,"l":0,"b":0})

fig.show()
# Grouping based on neighbourhood

bronx = airbnb['neighbourhood_group'] == 'Bronx'

staten_island = airbnb['neighbourhood_group'] == 'Staten Island'

queens = airbnb['neighbourhood_group'] == 'Queens'

brooklyn = airbnb['neighbourhood_group'] == 'Brooklyn'

manhattan = airbnb['neighbourhood_group'] == 'Manhattan'



# Calculating the avergae price in each neighbourhood

bronx_avg = airbnb[bronx]['price'].mean()

statenisland_avg = airbnb[staten_island]['price'].mean()

queens_avg = airbnb[queens]['price'].mean()

brooklyn_avg = airbnb[brooklyn]['price'].mean()

manhattan_avg = airbnb[manhattan]['price'].mean()





print("--- Average Price in the Neighbourhood Group ---")

print("Bronx:",bronx_avg)

print("Staten Island:",statenisland_avg)

print("Quuens:", queens_avg)

print("Brooklyn:",brooklyn_avg)

print("Manhattan:",manhattan_avg)
# Grouping based on room types

private_room = airbnb['room_type'] == 'Private room'

entire_home = airbnb['room_type'] == 'Entire home/apt'

shared_room = airbnb['room_type'] == 'Shared room'



# Calculating the avergae price in each neighbourhood

private_avg = airbnb[private_room]['price'].mean()

entire_avg = airbnb[entire_home]['price'].mean()

shared_avg = airbnb[shared_room]['price'].mean()





print("--- Average Price of the Room Types ---")

print("Private Room:",private_avg)

print("Entire Home/Apt:",entire_avg)

print("Shared room:", shared_avg)
# Most expensive Airbnbs in Brooklyn

expensive = airbnb[airbnb['neighbourhood_group'] == 'Brooklyn'].sort_values(by = 'price', ascending = False)

expensive.head(10)
# Visualizing the most used words in the names of the most expensive Airbnbs in Brooklyn

airbnb_brooklyn = airbnb[airbnb['neighbourhood_group'] == 'Brooklyn']

word_cloud = WordCloud(width = 1000,

                       height = 800,

                       colormap = 'GnBu', 

                       margin = 0,

                       max_words = 200,  

                       min_word_length = 4,

                       max_font_size = 120, min_font_size = 15,  

                       background_color = "white").generate(" ".join(airbnb_brooklyn['name']))



plt.figure(figsize = (10, 15))

plt.imshow(word_cloud, interpolation = "gaussian")

plt.axis("off")

plt.show()
expensive2 = airbnb[airbnb['neighbourhood_group'] == 'Manhattan'].sort_values(by = 'price', ascending = False)

expensive2.head(10)
# Visualizing the most used words in the names of the most expensive Airbnbs in Manhattan

airbnb_manhattan = airbnb[airbnb['neighbourhood_group'] == 'Manhattan']

word_cloud = WordCloud(width = 1000,

                       height = 800,

                       colormap = 'twilight_shifted', 

                       margin = 0,

                       max_words = 200,  

                       min_word_length = 4,

                       max_font_size = 120, min_font_size = 15,  

                       background_color = "white").generate(" ".join(airbnb_manhattan['name']))



plt.figure(figsize = (10, 15))

plt.imshow(word_cloud, interpolation = "gaussian")

plt.axis("off")

plt.show()
most_reviews = airbnb.sort_values(by = 'number_of_reviews', ascending = False)

most_reviews.head(10)
cheapest = airbnb.sort_values(by = 'price', ascending = True)

cheapest.head(10)
luxury = airbnb.sort_values(by = 'price', ascending = False)

luxury.head(10)
# Storing the names of the 100 cheapest airbnbs in New York in a separate variable



cheapest = cheapest.head(100)



# Visualizing the most used words in the names of the cheapest Airbnbs in New York City

word_cloud = WordCloud(width = 1000,

                       height = 800,

                       colormap = 'twilight_shifted', 

                       margin = 0,

                       max_words = 200,  

                       min_word_length = 4,

                       max_font_size = 120, min_font_size = 15,  

                       background_color = "white").generate(" ".join(cheapest['name']))



plt.figure(figsize = (10, 15))

plt.imshow(word_cloud, interpolation = "gaussian")

plt.axis("off")

plt.show()
# Storing the names of the 100 luxurious airbnbs in New York in a separate variable

luxury = luxury.head(100)



# Visualizing the most used words in the names of the cheapest Airbnbs in New York City

word_cloud = WordCloud(width = 1000,

                       height = 800,

                       colormap = 'twilight_shifted', 

                       margin = 0,

                       max_words = 200,  

                       min_word_length = 4,

                       max_font_size = 120, min_font_size = 15,  

                       background_color = "white").generate(" ".join(luxury['name']))



plt.figure(figsize = (10, 15))

plt.imshow(word_cloud, interpolation = "gaussian")

plt.axis("off")

plt.show()
# Selecting features for building the model

feature_columns = ['neighbourhood_group','room_type','price','minimum_nights', 'number_of_reviews',

                   'calculated_host_listings_count','availability_365']
# Visualizing the selected features

model_features = airbnb[feature_columns]

model_features.head(5)
# Separatign numerical and categorical features

numerical_features = model_features.dtypes[model_features.dtypes!="object"].index



print("Number of numerical features",len(numerical_features))

print(numerical_features)



#Pulling out names of categorical variables by conditioning dtypes

#Equal to object type



categorical_features = model_features.dtypes[model_features.dtypes=="object"].index

print("Number of categorical features",len(categorical_features))

print(categorical_features)
# Generating dummies for the categorical variables and concatenating them

neighbourhood_dummies = pd.get_dummies(model_features['neighbourhood_group'], drop_first = True)

room_dummies = pd.get_dummies(model_features['room_type'], drop_first = True)

all_data = pd.concat([model_features, neighbourhood_dummies, room_dummies ], axis = 1)

all_data.head(3)
# Removing the variables for which dummy values have been generated

all_data.drop('neighbourhood_group', axis = 1, inplace = True)

all_data.drop('room_type', axis = 1, inplace = True)

all_data.head(3)
# Splitting the data into train and test

y = all_data['price']

x = all_data.drop(['price'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 200)
# Building a linear regression model

lin_model = LinearRegression()

lin_model.fit(x_train,y_train)



# Making the predictions

y_pred = (lin_model.predict(x_test))