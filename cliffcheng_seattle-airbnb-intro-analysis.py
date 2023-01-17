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

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from decimal import Decimal

import datetime

import folium #Longitude and Lattitude mapping.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder #We don't this in the analysis I am currently showing.

from sklearn.linear_model import LinearRegression #We don't this in the analysis I am currently showing.

import seaborn as sns

from itertools import *

import os

import folium

from folium import plugins

from folium.plugins import MarkerCluster #To be able to cluster our individual data points on folium.

from IPython.display import HTML, display



calendar = pd.read_csv('../input/seattle/calendar.csv')

listing = pd.read_csv('../input/seattle/listings.csv')

reviews = pd.read_csv('../input/seattle/reviews.csv')
reviews.head()
calendar.head()
listing.head()
calendar = calendar.dropna(axis = 0, subset = ['price'], how = 'any')

reviews = reviews.dropna(axis = 0, subset = ['comments'], how = 'any')

calendar['date'] = pd.to_datetime(calendar['date'])

calendar['month'] = calendar.date.dt.month

calendar['year'] = calendar.date.dt.year

calendar['day'] = calendar.date.dt.day

calendar['price'] = pd.to_numeric(calendar['price'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')



listing['monthly_price'] = pd.to_numeric(listing['monthly_price'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')

listing['weekly_price'] = pd.to_numeric(listing['weekly_price'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')

listing['price'] = pd.to_numeric(listing['price'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')

listing['cleaning_fee'] = pd.to_numeric(listing['cleaning_fee'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')

listing['security_deposit'] = pd.to_numeric(listing['security_deposit'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')

listing['extra_people'] = pd.to_numeric(listing['extra_people'].apply(lambda x: str(x).replace('$', '').replace(',', '')), errors='coerce')

listing = listing.rename(columns = {'id':'listing_id'})
(listing.isnull().sum()[listing.isnull().sum().nonzero()[0]])/len(listing) 
default_list = listing[['property_type', 'neighbourhood', 'review_scores_value', 

                        'bathrooms', 'bedrooms', 'price', 'longitude', 'latitude']]



new_list = default_list.dropna(axis = 0, how = 'any')

new_list
def Plot(cur, data_list):

    """Description: This function can be used to plot a graph by reading the data_list and grouping by the cursor obejct.

    

    Arguments: 

    cur: the cursor object.

    data_list: list of data.

    

    Returns:

    A graphical reprementation of the cur items in the data_list."""





    plt.figure(figsize=(20,20))

    plt.xticks(rotation=90)

    sns.countplot((data_list)[(cur)],

                 order = data_list[cur].value_counts().index)

    plt.show()
(new_list.property_type.value_counts())/(new_list.property_type.count())
Plot('property_type', new_list)
(new_list.neighbourhood.value_counts())/(new_list.neighbourhood.count())
Plot('neighbourhood', new_list)
new_list_neighbourhood = new_list.groupby('neighbourhood').count()

new_list_top_15_neighbourhood = new_list_neighbourhood.nlargest(35,'property_type')

new_list_top_15_neighbourhood



neighbourhood_list = ['Capitol Hill',

'Ballard',

'Belltown',

'Minor',

'Queen Anne',

'Fremont',

'Wallingford',

'First Hill',

'North Beacon Hill',

'University District',

'Stevens',

'Central Business District',

'Lower Queen Anne',

'Greenwood',

'Columbia City',

'Ravenna',

'Magnolia',

'Atlantic',

'North Admiral',

'Phinney Ridge',

'Green Lake',

'Leschi',

'Mount Baker',

'Eastlake',

'Maple Leaf',

'Madrona',

'Pike Place Market',

'The Junction',

'Seward Park',

'Bryant',

'Genesee',

'North Delridge',

'Roosevelt',

'Crown Hill',

'Montlake']
#This gives me a list of True/False statements for each row if the value in the neighbourhood columns is in 

#neighbourhood_list above.

true_false_by_neighbourhood = new_list.neighbourhood.isin(neighbourhood_list) 



#I can then put this new list of True/False statements into our origional new_list. This filters new_list down

# to 35 categories of the neighbourhood column while still containing 86.72% of the origional data.

filtered_neighborhood = new_list[true_false_by_neighbourhood]

filtered_neighborhood
new_list_property_type = new_list.groupby('property_type').count()

new_list_top_16_property_type = new_list_property_type.nlargest(16, 'neighbourhood')

new_list_top_16_property_type





property_type_list = ['House', 'Apartment', 'Townhouse', 'Condominium', 'Loft', 'Bed & Breakfast', 'Cabin']
#This gives me a list of True/False statements for each row if the value in the property_type column is in 

#the property_type_list above.

true_false_by_property = filtered_neighborhood.property_type.isin(property_type_list) 



#I can then put this new list of True/False statements into our filtered_neighbourhood list. 

#This filters new_list down seven property types while still containing 98.30% of the data of filtered_neighbourhood.

filtered_data = filtered_neighborhood[true_false_by_property]

filtered_data
sns.lmplot(data=filtered_data, x='bedrooms', y='price', hue='review_scores_value')
room_premium = (filtered_data.price)/(filtered_data.bedrooms)

filtered_data['Cost Per Bedroom'] = filtered_data['latitude'].add(room_premium)

filtered_data
def Filterlist(cur, filepath):

    """Description: This function can be used to read the file and filter based on the input.

    

    Arguments: 

    cur: the cursor object.

    filepath: data file

    

    Returns: 

    The data file that is filtered by the cur object."""

    

    property_type = [cur]

    true_false_by_property = filtered_data.property_type.isin(property_type) 

    List = filtered_data[true_false_by_property]

    return List



House_list = Filterlist('House', filtered_data)

House_list
def catplot(x_data, y_data, data_list):

    """Description: This function can be used to read the file in the data_list and create a catplot based on the

    x_data and y_data.

    

    Arguments: 

    x_data: Column in data_list that you want to plot on the x axis (Put as string). 

    y_ data: Column in the data_list that you want to plot on the y axis (Put as string).

    data_list: The datalist that contains the x_data and y_data as columns."""

    sns.catplot(x=x_data, y=y_data, data=data_list, height=8)
catplot('Cost Per Bedroom', 'neighbourhood', House_list)
Apartment_list = Filterlist('Apartment', filtered_data)

Apartment_list
catplot('Cost Per Bedroom', 'neighbourhood', Apartment_list)
def folium_plot(x, y, data_list):

    """Find the latitude and longitude columns in data_list and plots them. 

    Then it custers the data into groups to be viewed on different levels.

    

    Arguements: 

        latitude: columns containing latitude coordinates and labeled as latitude. Write as string.

        longitude: columns containing longitude coordinates and labeled as longitude. Write as string.

        data_list: Data list where columns are held.

    

    Returns:

    Clustered map of all data points."""

    

    #Creates a map of Seattle.

    m = folium.Map(location=[47.60, -122.24], zoom_start = 11)

    m.save('index.html')





    #Takes the latitude and longitude coordinates and zips them into a form to be plotted.

    lat = pd.to_numeric(data_list[x], errors = 'coerce')

    lon = pd.to_numeric(data_list[y], errors = 'coerce')



    #Zip togethers each list of latitude and longitude coordinates. 

    result = zip(lat,lon)

    lat_lon = list(result)





    mc = MarkerCluster().add_to(m)

    for i in range(0,len(data_list)):

        folium.Marker(location=lat_lon[i]).add_to(mc)



    m.save('index.html')

    display(m)
folium_plot('latitude', 'longitude', House_list)
folium_plot('latitude', 'longitude', Apartment_list)