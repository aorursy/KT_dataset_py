# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from matplotlib import pyplot as plt

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df = pd.read_csv('/kaggle/input/bournemouth-venues/bournemouth_venues.csv')

# Any results you write to the current directory are saved as output.
print(df.head())
venues = df['Venue Category'] # variable so I don't have to type as much later

#  Lists of Categories for mapping the new column "Venue General Categories"

restaurant_extras = ['Sandwich Place', 'Diner', 'Pizza Place', 'Noodle House', 'Burger Joint', 'Indian Restaurant', 'English Restaurant', 'Fast Food Restaurant', 'French Restaurant']

cafe_extras = ['Coffee Shop', 'Ice Cream Shop', 'Café', 'Bubble Tea Shop', 'Dessert Shop']

bar_extras = ['Pub', 'Nightclub', 'Brewery', ]

indoor_recreation_extras = ['Multiplex', 'Theater', 'Arts & Entertainment']

outdoor_recreation_extras = ['Park', 'Plaza', 'Beach', 'Garden', 'Other Great Outdoors', 'Scenic Lookout']

educational_extras = ['Art Museum', 'Aquarium']

retail_extras = ['Clothing Store', 'Grocery Store']

transit_extras = ['Train Station', 'Bus Stop', 'Platform']
#  Creating individual dataframes for each group to be used in FacetGrid()

restaurant_data_1 = df[venues.isin(restaurant_extras)]

restaurant_data_2 = df[venues.str.contains('Restaurant')]

restaurant_data_2 = restaurant_data_2[~restaurant_data_2.isin(restaurant_data_1)].dropna() # The '~' symbol flips the booleans

cafe_data = df[venues.isin(cafe_extras)]

bar_data = df[(venues.str.contains('Bar')) | (venues.isin(bar_extras))]

indoor_recreation_data = df[venues.isin(indoor_recreation_extras)]

outdoor_recreation_data = df[venues.isin(outdoor_recreation_extras)]

educational_data = df[venues.isin(educational_extras)]

retail_data = df[venues.isin(retail_extras)]

gym_data = df[venues.str.contains('Gym')]

transit_data = df[venues.isin(transit_extras)]

hotel_data = df[venues.str.contains('Hotel')]
#  Prepping variables for looping

dataframe_list = [restaurant_data_1, restaurant_data_2, cafe_data, bar_data, indoor_recreation_data, outdoor_recreation_data,

         educational_data, retail_data, gym_data, transit_data, hotel_data]

dataframe_names = ['Restaurant Group 1', 'Restaurant Group 2', 'Cafe', 'Bar', 'Indoor Recreation', 'Outdoor Recreation', 'Educational',

                  'Retail', 'Gym', 'Transit', 'Hotel']



#  Check that the lists are of equal length

if len(dataframe_list) - len(dataframe_names) != 0:

    print('ERROR: Number of dataframes is not equal to number of dataframe names')
#  Rename rows with the General Category Names using a for loop

df['Venue General Category'] = df['Venue Category']

for i in range(len(dataframe_list)):

    df['Venue General Category'] = df.apply(lambda row: dataframe_names[i] \

                                        if row['Venue General Category'] in (list(dataframe_list[i]['Venue Category'])) \

                                        else row['Venue General Category'], \

                                        axis=1)

#print(df['Venue General Category'].head(10))



#  Checking to make sure I didn't miss any categories

#frames = [restaurant_data, cafe_data, bar_data, indoor_recreation_data, outdoor_recreation_data,

#         educational_data, retail_data, gym_data, transit_data, hotel_data]

#all_data = pd.concat(frames)



#list1 = all_data['Venue Category'].unique()

#list2 = df['Venue Category'].unique()

#print(list(set(list2) - set(list1)))

def scatter_plot_facetgrid_by_category(data, x_col, y_col, color_by):

    sns.set(style='whitegrid')

    f, ax = plt.subplots(figsize=(20, 60))

    for i in range(len(dataframe_list)):

        plt.subplot(6, 2, i+1)

        plt.title(dataframe_names[i])

        g = sns.scatterplot(data=dataframe_list[i], x=x_col, y=y_col, hue=color_by, legend='full', palette='Paired', s=150)

        g.set(xlim=(min(df[x_col]), max(df[x_col])), ylim=(min(df[y_col]), max(df[y_col])))
scatter_plot_facetgrid_by_category(data=df, x_col='Venue Longitude', y_col='Venue Latitude', 

                               color_by='Venue Category')

plt.show()