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
import matplotlib.pyplot as plt

import datetime as dt

pd.options.mode.chained_assignment = None  # default='warn'
#create new variables for the datasets for ease of referencing throughout the project.

seattle = pd.read_csv("../input/restaurant-inspection/Food_Establishment_Inspection_Data.csv")

chicago = pd.read_csv("../input/restaurant-inspection/Food_Inspections.csv")
seattle.head()
chicago.tail()
#add a new column based on Violation Type so it's more descriptive and meaningful to the user

seattle['Violation Cat'] = np.where(seattle['Violation Type']=='BLUE', 'Maintenance & Sanitation', 'Food Handling Practices')



#add a new column based on Grade so it's more descriptive and meaningful to the user



grade_descrip = [] #create a list



for row in seattle['Grade']:

    if   row > 3.0:  grade_descrip.append('Needs Improvement')

    elif row > 2.0:  grade_descrip.append('Okay')

    elif row > 1.0:  grade_descrip.append('Good')

    elif row > 0:  grade_descrip.append('Excellent')

    else:          grade_descrip.append('No Grade')

    # Create a column from the list

    

seattle['Grade Descrip.'] = grade_descrip #append the list as a new column

seattle.head() #inspect 


seattle1 = seattle[['Program Identifier','Inspection Date','Inspection Score','Inspection Result','Grade','Grade Descrip.','Violation Type','Violation Cat','Violation Description','Longitude','Latitude']]
seattle1.head()
#filter to just Restaurants first

chicago_filter = chicago[(chicago['Facility Type'] == 'Restaurant')]

chicago1 = chicago_filter [['AKA Name','Inspection Date','Risk','Results','Violations','Longitude','Latitude']]

chicago1.head()
#Create a new column "Violation Code" baed on "Violations"

chicago1.loc[chicago1['Violations'].notnull(),'Violation Code'] = chicago1['Violations'].str[:2]

chicago1.head()
#Seattle Dataset 

#First see what the insepction result distribution is

seattle1['Inspection Result'].value_counts().plot.pie(autopct='%1.1f%%')



# Unsquish the pie.

plt.gca().set_aspect('equal')
#Then see what the Grade distribution is 

seattle1['Grade Descrip.'].value_counts().plot.pie(autopct='%1.1f%%')

# Unsquish the pie.

plt.gca().set_aspect('equal')
chicago1['Results'].value_counts().plot.pie(autopct='%1.1f%%')



# Unsquish the pie.

plt.gca().set_aspect('equal')
pd.options.mode.chained_assignment = None  # default='warn'

code_seattle1 = seattle1[seattle1['Violation Type'].notnull()]

code_seattle1['Inspection Date']= pd.to_datetime(code_seattle1['Inspection Date']) #convert inspection date

code_seattle1.sort_values(by='Inspection Date')

# plot line chart

code_seattle1['year'] = pd.DatetimeIndex(code_seattle1['Inspection Date']).year

code_seattle1['year'] = code_seattle1['year'].apply(pd.to_numeric) #change data type to numeric

code_seattle1.sort_values(by='year')

code_seattle1.head()



    
stacked_bar = code_seattle1.groupby(['year', 'Violation Cat'])['year'].count().unstack('Violation Cat').fillna(0)

stacked_bar[['Maintenance & Sanitation','Food Handling Practices']].plot(kind='bar', stacked=False)

plt.title('Seattle Violation Types')

plt.ylabel('Count')
code_chicago1 = chicago1[chicago1['Violation Code'].notnull()] #remove null values

code_chicago1['Violation Code'] = code_chicago1['Violation Code'].str.replace('.', '') #take out periods from the column

code_chicago1['Violation Code'] = code_chicago1['Violation Code'].apply(pd.to_numeric) #change data type to numeric



code_chicago1.head() #inspect 
#like the Seattle dataset, create a new column called "Violation Cat" based on the Violation code number.

# This groups the violation groups into 3 different categories, based on Chicago City's Standards



violation_cat = [] #create a list



for row in code_chicago1['Violation Code']:

    if row > 29:    violation_cat.append('Minor')

    elif row > 14:  violation_cat.append('Serious')

    elif row > 0:   violation_cat.append('Critical')

    else:           violation_cat.append('ERROR')

    # Create a column from the list

    

code_chicago1['Violation Cat'] = violation_cat #append the list as a new column

code_chicago1.head() #inspect 
#convert inspection Date to datetime

code_chicago1['Inspection Date']= pd.to_datetime(code_chicago1['Inspection Date']) #convert inspection date

code_chicago1.sort_values(by='Inspection Date')

#create a new column "year"

code_chicago1['year'] = pd.DatetimeIndex(code_chicago1['Inspection Date']).year

code_chicago1['year'] = code_chicago1['year'].apply(pd.to_numeric) #change data type to numeric

code_chicago1.sort_values(by='year')

code_chicago1.head() #inspect
stacked_bar1 = code_chicago1.groupby(['year', 'Violation Cat'])['year'].count().unstack('Violation Cat').fillna(0)

stacked_bar1[['Critical','Minor','Serious']].plot(kind='bar', stacked=False)

plt.title('Chicago Violation Types')

plt.ylabel('Count')
#import necessary packages

import descartes

import geopandas as gpd

import shapefile as shp

from shapely.geometry import Point, Polygon
crs = {'init': 'espg:4326'}

geometry = [Point(xy) for xy in zip (code_seattle1['Longitude'], code_seattle1['Latitude'])]
geo_seattle = gpd.GeoDataFrame(code_seattle1, crs = crs, geometry = geometry)

geo_seattle.head()
street_map = gpd.read_file("../input/king-county-streets") #retrieve files
#plot on map

fig,ax = plt.subplots(figsize = (25,30))

street_map.plot (ax = ax, alpha = 0.2,)

geo_seattle[geo_seattle['Violation Cat'] == 'Maintenance & Sanitation'].plot (ax = ax, markersize = 2, color = 'blue', marker = 'o',label = "Maint")

geo_seattle[geo_seattle['Violation Cat'] == 'Food Handling Practices'].plot (ax = ax, markersize = 2, color = 'red', marker = 'o',label = "Food Hand")

plt.legend(prop = {'size': 15})

plt.title('Seatle Distribution of Violations', size = 30)
crs = {'init': 'espg:4326'}

geometry_chicago = [Point(xy) for xy in zip (code_chicago1['Longitude'], code_chicago1['Latitude'])]
geo_chicago = gpd.GeoDataFrame(code_chicago1, crs = crs, geometry = geometry_chicago)

geo_chicago.head()
street_map1 = gpd.read_file("../input/chicago")
fig,ax = plt.subplots(figsize = (25,30))

street_map1.plot (ax = ax, alpha = 0.2,)

geo_chicago[geo_chicago['Violation Cat'] == 'Critical'].plot (ax = ax, markersize = 2, color = 'red', marker = 'o',label = "Critical")

geo_chicago[geo_chicago['Violation Cat'] == 'Serious'].plot (ax = ax, markersize = 2, color = 'blue', marker = 'o',label = "Serious")

geo_chicago[geo_chicago['Violation Cat'] == 'Minor'].plot (ax = ax, markersize = 2, color = 'green', marker = 'o',label = "Minor")

plt.legend(prop = {'size': 15})

plt.title('Chicago Distribution of Violations', size = 30)