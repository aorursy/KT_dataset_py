# Install using conda/pip if the following libraries are not installed by uncommenting them

    # ! pip install tqdm

    # ! pip install pandas

    # ! pip install numpy

    # ! pip install matplotlib

    # ! pip install seaborn

    # ! pip install plottly

    # ! pip install cufflinks

# Importing libraries

import os

import shutil

import math

import numpy as np

import pandas as pd

import seaborn as sns

import cufflinks as cf

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt

from tqdm import tqdm

from plotly.subplots import make_subplots

from collections import Counter

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
# Getting to know about our dataset

data_raw = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')

data_raw.head()
# Let us get rid of unwanted columns

data_raw.drop(['url', 'phone', 'menu_item', 'reviews_list', 'dish_liked'], axis=1, inplace=True)

data_raw.head()
# Understanding the structure

data_raw.info()
# Checking for null values

data_raw.isna().any()
# First - working on 'cuisines' column. Converting the object into a list so that it becomes easier for us to analyse

data_raw['cuisines'] = data_raw['cuisines'].apply(lambda x: x.split(',') if type(x) is str else "Nill")

# Second - working on 'rest_types' column. Converting the object into a list so that it becomes easier for us to analyse

data_raw['rest_type'] = data_raw['rest_type'].apply(lambda x: x.split(',') if type(x) is str else "Nill")

# Third - filling and cleaning 'location' columns by using 'address' columns, by doing this we can get rid of 'address' column and use just the 'location' column.

for record_num in range(data_raw.shape[0]):

    if type(data_raw['location'][record_num]) is float:

        data_raw['location'][record_num] = data_raw['address'][record_num].split(', ')[-2]

    elif len(data_raw['location'][record_num].split(', ')) > 1:

        data_raw['location'][record_num] = data_raw['location'][record_num].split(', ')[-1]

data_raw.drop(['address'], axis=1, inplace=True)
# Function to get counts for cuisines

def get_cuisines_count_plot(cuisines_col):

    # Getting the count of every item

    counter = Counter()

    for i in cuisines_col:

        for j in i:

            counter[j] += 1

    # Filtering counts and selecting everything above 1000 and plotting

    x_axis, y_axis = [], []

    for key, value in counter.items():

        if value >= 500:

            x_axis.append(key)

            y_axis.append(value)

    fig = plt.figure(figsize=(12, 5))

    axes = fig.add_axes([0.1, 0.1, 1.0, 0.8])

    axes.set_xlabel('Counts')

    axes.set_ylabel('Cuisines')

    axes.set_title('Cuisines prefered in Bangalore')

    axes.bar(x_axis, y_axis, color=(0.5, 0.5, 0.5, 0.1),  edgecolor='blue')

    plt.xticks(rotation=90)

    return counter



# Function to get counts for restaurant types

def get_rest_type_count_plot(rest_types):

    # Getting the count of every item

    counter = Counter()

    for i in rest_types:

        for j in i:

            counter[j] += 1

    # Filtering counts and selecting everything above 1000 and plotting

    x_axis, y_axis = [], []

    for key, value in counter.items():

        if value >= 500:

            x_axis.append(key)

            y_axis.append(value)

    fig = plt.figure(figsize=(12, 5))

    axes = fig.add_axes([0.1, 0.1, 1.0, 0.8])

    axes.set_xlabel('Counts')

    axes.set_ylabel('Restaurant types')

    axes.set_title('Restaurant types prefered in Bangalore')

    axes.bar(x_axis, y_axis, color=(0.5, 0.5, 0.5, 0.1),  edgecolor='blue')

    plt.xticks(rotation=45)

    return counter



# Function to check stats for restaurants offering online delivery and table booking facility

def get_facility_count(data):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    sns.countplot(x='online_order', data=data, orient='v', ax=axes[0])

    axes[0].set_title('Restaurants with online delivery facility in Bangalore')

    sns.countplot(x='book_table', data=data, orient='v', ax=axes[1])

    axes[1].set_title('Restaurants with table reservation facility in Bangalore')

    plt.tight_layout()

    return
# Visualizing data

cuisine_counts = get_cuisines_count_plot(data_raw['cuisines'])

rest_type_counts = get_rest_type_count_plot(data_raw['rest_type'])

get_facility_count(data_raw)
# Analysing the data further

data_layer_indices = list(zip(data_raw['location'], data_raw['name']))

data_index = pd.MultiIndex.from_tuples(data_layer_indices)

rating_votes = np.concatenate((np.array([data_raw['online_order']]).T, np.array([data_raw['book_table']]).T, np.array([data_raw['rest_type']]).T, np.array([data_raw['cuisines']]).T, np.array([data_raw['approx_cost(for two people)']]).T, np.array([data_raw['rate']]).T, np.array([data_raw['votes']]).T),  axis=1)

data = pd.DataFrame(rating_votes, data_index, columns=['online_order', 'book_table', 'rest_type', 'cuisines', 'approx_cost_for_2_people', 'rate', 'votes'])

data

# It looks like a particular restaurant is repeated many times in our data. We dont want this. The reason for this might be - gathering of information on different days.

# Uncomment the below line to see the proof

# data.loc['Whitefield'].loc['Jalsa']
# Checkpoint -> saving the updated data, if you want

# data_raw.to_csv('zomato_updated.csv')

# Here, we remove such entities (above cell)

unique_location_list = list(data_raw['location'].unique())

for i in tqdm(unique_location_list):

    for j in list(data_raw[data_raw['location'] == i]['name']):

        max_votes_for_a_particular_restaurant = data_raw[(data_raw['location'] == i) & (data_raw['name'] == j)]['votes'].max()

        temp = data_raw[(data_raw['location'] == i) & (data_raw['name'] == j) & (data_raw['votes'] < max_votes_for_a_particular_restaurant)]

        data_raw.drop(temp.index, inplace=True)

data_raw.drop_duplicates(subset=['name', 'location'], keep='first', inplace=True)
# Reading the updated data and visulaizing it

data_updated = data_raw

cuisine_counts = get_cuisines_count_plot(data_updated['cuisines'])

rest_type_counts = get_rest_type_count_plot(data_updated['rest_type'])

get_facility_count(data_updated)
# Analysing the data after cleaning

data_layer_indices = list(zip(data_updated['location'], data_updated['name']))

data_index = pd.MultiIndex.from_tuples(data_layer_indices)

rating_votes = np.concatenate((np.array([data_updated['online_order']]).T, np.array([data_updated['book_table']]).T, np.array([data_updated['rest_type']]).T, np.array([data_updated['cuisines']]).T, np.array([data_updated['approx_cost(for two people)']]).T, np.array([data_updated['rate']]).T, np.array([data_updated['votes']]).T),  axis=1)

data = pd.DataFrame(rating_votes, data_index, columns=['online_order', 'book_table', 'rest_type', 'cuisines', 'approx_cost_for_2_people', 'rate', 'votes'])

data.loc['Banashankari']
# Visualization for the selected area

def sunburst_plot(data, loc):

    # Visualizing restaurants by location based on the number of votes, top 20 items

    top_items = 20

    location = loc

    character = [location]

    parent = ['']

    values = [top_items]

    temp = data.loc[location].sort_values(by=['votes', 'rate'], ascending=False)[: top_items]

    hotel_names = list(temp.index)

    for i in hotel_names:

        character.append(i)

        parent.append(location)

        character.append('Cuisines in ' + i + ' -> ' + str(temp.loc[i].cuisines))

        parent.append(i)

        character.append(i + ' type is -> ' + str(temp.loc[i].rest_type))

        parent.append(i)

        character.append('Online ordering facility in ' + i + ' -> ' + temp.loc[i].online_order)

        parent.append(i)

        character.append('Table booking facility ' + i + ' -> ' + temp.loc[i].book_table)

        parent.append(i)

        character.append('Approximate bill for two people ' + i + ' ->'  + str(temp.loc[i].approx_cost_for_2_people))

        parent.append(i)

    fig = go.Figure(go.Sunburst(labels=character, parents=parent))

    fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))

    fig.show()

    return



def pie_chart(data, loc):

    # Note that for categories 'Others', number of votes = number of restaurants not popular (< 500 votes)

    temp = data.loc[loc]

    temp['name'] = 'Others'

    hotel_names = list(temp.index)

    for c, i in enumerate(hotel_names):

        if temp.loc[i].votes > 500:

            temp.at[i, 'name'] = i

        else:

            temp.at[i, 'votes'] = 1

    fig = px.pie(temp, values='votes', names='name', title='Popularity of a restaurant')

    fig.show()

    return
# location = input("Enter a location ->")

location = 'BTM'

sunburst_plot(data, location)

pie_chart(data, location)