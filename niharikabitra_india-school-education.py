# Libraries for storage and processing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Libraries to create maps
import folium 
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from geopy.geocoders import Nominatim

pd.options.mode.chained_assignment = None
import geopandas as gpd

# Plotting packages
import seaborn as sns
import matplotlib.pyplot as plt
# Read data
dropout = pd.read_csv('/kaggle/input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')
enrollment = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')
computers = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-comps-2013-2016.csv')
electricity = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-electricity-2013-2016.csv')
water = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-water-facility-2013-2016.csv')
boys_toilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-boys-toilet-2013-2016.csv')
girls_toilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-girls-toilet-2013-2016.csv')
# Number of columns and rows
print('Shape:', dropout.shape)

# Data Preview
dropout
# Data Types
dropout.dtypes
# Check the unique values of Indian States
print('Number of Indian States: ', dropout.State_UT.nunique())
print('List of Indian States:\n', dropout.State_UT.unique())
# Replace multiple spaces with 1 space for Indian States
dropout['State_UT'] = dropout.State_UT.str.replace(' +', ' ')

# Update shortened version of a state to its full name
dropout['State_UT'] = dropout.State_UT.str.replace('A & N Islands', 'Andaman and Nicobar Islands')

# Update ampersand to the word 'and'
dropout['State_UT'] = dropout.State_UT.str.replace('&', 'and')

print('Number of Indian States: ', dropout.State_UT.nunique())
print('List of Indian States:\n', dropout.State_UT.unique())
# Create a list of Indian States for reference
indian_states = dropout.State_UT.unique()
# Check the unique values of Indian States
print('Number of Years: ', dropout.year.nunique())
print('List of Unique Years:\n', dropout.year.unique())
# Rename column names
new_column_names = {
    'State_UT' : 'State',
    'year' : 'Year',
    'Upper Primary_Boys': 'Upper_Primary_Boys',
    'Upper Primary_Girls': 'Upper_Primary_Girls',
    'Upper Primary_Total' : 'Upper_Primary_Total',
    'Secondary _Boys' : 'Secondary_Boys',
    'Secondary _Girls' : 'Secondary_Girls',
    'Secondary _Total' : 'Secondary_Total',
    'HrSecondary_Boys' : 'Hr_Secondary_Boys',
    'HrSecondary_Girls' : 'Hr_Secondary_Girls',
    'HrSecondary_Total' : 'Hr_Secondary_Total'
}
dropout = dropout.rename(columns=new_column_names)

# Convert data types for specific columns 
# For invalid values, replace to NaN. In this case, there are  multiple NR values.
dropout['Primary_Girls'] = pd.to_numeric(dropout['Primary_Girls'], errors='coerce')
dropout['Primary_Boys'] = pd.to_numeric(dropout['Primary_Boys'], errors='coerce')
dropout['Primary_Total'] = pd.to_numeric(dropout['Primary_Total'], errors='coerce')
dropout['Upper_Primary_Boys'] = pd.to_numeric(dropout['Upper_Primary_Boys'], errors='coerce')
dropout['Upper_Primary_Girls'] = pd.to_numeric(dropout['Upper_Primary_Girls'], errors='coerce')
dropout['Upper_Primary_Total'] = pd.to_numeric(dropout['Upper_Primary_Total'], errors='coerce')
dropout['Secondary_Boys'] = pd.to_numeric(dropout['Secondary_Boys'], errors='coerce')
dropout['Secondary_Girls'] = pd.to_numeric(dropout['Secondary_Girls'], errors='coerce')
dropout['Secondary_Total'] = pd.to_numeric(dropout['Secondary_Total'], errors='coerce')
dropout['Hr_Secondary_Boys'] = pd.to_numeric(dropout['Hr_Secondary_Boys'], errors='coerce')
dropout['Hr_Secondary_Girls'] = pd.to_numeric(dropout['Hr_Secondary_Girls'], errors='coerce')
dropout['Hr_Secondary_Total'] = pd.to_numeric(dropout['Hr_Secondary_Total'], errors='coerce')
# Set the index
dropout = dropout.set_index(['State', 'Year'])
dropout.index.is_unique
dropout.dtypes
dropout.describe(include='all')
boxplot = dropout.boxplot(column=['Primary_Boys', 'Primary_Girls'])
india_map = folium.Map(location=[20.5937,78.9629], tiles='cartodbpositron', zoom_start=5)
geolocator = Nominatim(user_agent='my_app')

def evaluator(val):
    if val < 10:
        return 'green'
    elif val < 20:
        return 'orange'
    else:
        return 'red'

# Add points to the map
for idx, row in dropout.iterrows():
    state_name = idx[0]
    coordinate = geolocator.geocode(state_name)
    
    Marker([coordinate.latitude, coordinate.longitude], popup=[state_name, row['Primary_Girls']], icon=folium.Icon(color=evaluator(row['Primary_Girls']))).add_to(india_map)

# Display the map
india_map


