import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib.cm
%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/statewise-motorvehicles/datafile.csv")
df.head()
df['States/UTs'] = df['States/UTs'].str.replace('&','and')
vehicle_1314 = []
vehicle_1415 = []
for i in range(len(df)):
    states = df.iloc[i,0]
    rate_1314 = df.iloc[i,1]
    rate_1415 = df.iloc[i,2]
    vehicle_1314.append((states,rate_1314))
    vehicle_1415.append((states,rate_1415))
    
print ("No. of Motor Vehicles per 1000 population - During 2013-14 - \n ",vehicle_1314)
print("\n No. of Motor Vehicles per 1000 population - During 2014-15 - \n ", vehicle_1415)
from mpl_toolkits.basemap import Basemap
'''
STEP 2 : CREATING A MAP
'''
# Create figure 
fig, ax = plt.subplots()
# Create a map with the coordinates determined by the Bounding Box tool
m = Basemap(projection='merc',lat_0=54.5, lon_0=-4.36,llcrnrlon=68.1, llcrnrlat= 6.5, urcrnrlon=97.4, urcrnrlat=35.5)
# Draw map boundary and set the color
m.drawmapboundary(fill_color='#46bcec')
# Fill continents and lakes
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
# Draw coast lines
m.drawcoastlines()

'''
USING SHAPEFILES FOR DRAWING STATES 
'''
# Load the shape file of India

m.readshapefile(
    "../input/shapefile/india_shape/India_Shape/india_st","INDIA")



'''
STEP 2 : CREATING A MAP
'''
# Create figure 
fig, ax = plt.subplots()
# Create a map with the coordinates determined by the Bounding Box tool
m = Basemap(projection='merc',lat_0=54.5, lon_0=-4.36,llcrnrlon=68.1, llcrnrlat= 6.5, urcrnrlon=97.4, urcrnrlat=35.5)
# Draw map boundary and set the color
m.drawmapboundary(fill_color='#46bcec')
# Fill continents and lakes
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
# Draw coast lines
m.drawcoastlines()

'''
USING SHAPEFILES FOR DRAWING STATES 
'''
# Load the shape file of India

m.readshapefile(
    "../input/shapefile/india_shape/India_Shape/india_st","INDIA")



# Create an empty list to hold number of vehicles in 2013-14
Num_of_vehicles_1314 = []

for state_info in m.INDIA_info:
    # Get the state in uppercase, as our csv file has state names in upper case
    state = state_info['STATE'].upper()
    # initialize number of vehicles = 0
    num = 0
    
    # In vehicle_1314 (containing tuples of state name and no. of vehicles), search for state 'state'
    # Append its corresponding number to Num_of_vehicles_1314
    for x in vehicle_1314:

        if x[0].upper() == state:
            num = x[1]
            break
    Num_of_vehicles_1314.append(num) 
    
    
# Create a dataframe containing shapes, state names and no. of vehicles   
df_poly = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in m.INDIA],
        'area': [area['STATE'] for area in m.INDIA_info],
        'Num_of_vehicles_1314' : Num_of_vehicles_1314
    })




plt.rcParams['figure.figsize'] = (30,30)
plt.rcParams.update({'font.size': 10})
'''
STEP 2 : CREATING A MAP
'''
import matplotlib.pyplot as plt
# Create figure 
fig, ax = plt.subplots()
# Create a map with the coordinates determined by the Bounding Box tool
m = Basemap(projection='merc',lat_0=54.5, lon_0=-4.36,llcrnrlon=68.1, llcrnrlat= 6.5, urcrnrlon=97.4, urcrnrlat=35.5)
# Draw map boundary and set the color
m.drawmapboundary(fill_color='#46bcec')
# Fill continents and lakes
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
# Draw coast lines
m.drawcoastlines()

'''
USING SHAPEFILES FOR DRAWING STATES 
'''
# Load the shape file of India
m.readshapefile(
    "../input/shapefile/india_shape/India_Shape/india_st","INDIA")




# Create an empty list to hold number of vehicles in 2013-14
Num_of_vehicles_1314 = []

for state_info in m.INDIA_info:
    # Get the state in uppercase, as our csv file has state names in upper case
    state = state_info['STATE'].upper()
    # initialize number of vehicles = 0
    num = 0
    
    # In vehicle_1314 (containing tuples of state name and no. of vehicles), search for state 'state'
    # Append its corresponding number to Num_of_vehicles_1314
    for x in vehicle_1314:

        if x[0].upper() == state:
            num = x[1]
            break
    Num_of_vehicles_1314.append(num) 
    
    
# Create a dataframe containing shapes, state names and no. of vehicles   
df_poly = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in m.INDIA],
        'area': [area['STATE'] for area in m.INDIA_info],
        'Num_of_vehicles_1314' : Num_of_vehicles_1314
    })
 
# Get all the shapes
shapes = [Polygon(np.array(shape), True) for shape in m.INDIA]
# Create a colormap
cmap = plt.get_cmap('Oranges')   
# Create a patch collection. Create patches on the top of the map, not beneath it (zorder=2)
pc = PatchCollection(shapes, zorder=2)

norm = Normalize()
# Set color according to the number of vehicle of the state
pc.set_facecolor(cmap(norm(df_poly['Num_of_vehicles_1314'].fillna(0).values)))
ax.add_collection(pc)

printed_names = []
for shapedict,state in zip(m.INDIA_info, m.INDIA):
    state_name = [value for key, value in shapedict.items()][0]
    
    if state_name in printed_names: continue
    # center of polygon
    x, y = np.array(state).mean(axis=0)
    # You have to align x,y manually to avoid overlapping for little states
    plt.text(x+.1, y, state_name, ha="center")
    printed_names += [state_name,] 


# Create a mapper to map color intensities to values
mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
mapper.set_array(Num_of_vehicles_1314)
plt.colorbar(mapper, shrink=0.4)
# Set title for the plot
ax.set_title("NUMBER OF VEHICLES OF INDIAN STATES FOR THE YEAR 2013-2014")
# Change plot size and font size
#plt.rcParams['figure.figsize'] = (30,30)
# plt.figure(figsize=(30,30))
# plt.rcParams.update({'font.size': 10})
plt.show()
'''
STEP 2 : CREATING A MAP
'''
# Create figure 
fig, ax = plt.subplots()
# Create a map with the coordinates determined by the Bounding Box tool
m = Basemap(projection='merc',lat_0=54.5, lon_0=-4.36,llcrnrlon=68.1, llcrnrlat= 6.5, urcrnrlon=97.4, urcrnrlat=35.5)
# Draw map boundary and set the color
m.drawmapboundary(fill_color='#46bcec')
# Fill continents and lakes
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
# Draw coast lines
m.drawcoastlines()

'''
STEP 3 : USING SHAPEFILES FOR DRAWING STATES 
'''
# Load the shape file of India
m.readshapefile(
    "../input/shapefile/india_shape/India_Shape/india_st","INDIA")


'''
STEP 4 : CREATING A DATAFRAME MAPPING SHAPES TO STATE NAME AND NUMBER OF VEHICLES
'''
# Create an empty list to hold number of vehicles in 2014-15
Num_of_vehicles_1415 = []

for state_info in m.INDIA_info:
    # Get the state in uppercase, as our csv file has state names in upper case
    state = state_info['STATE'].upper()
    # initialize number of vehicles = 0
    num = 0
    
    # In vehicle_1314 (containing tuples of state name and no. of vehicles), search for state 'state'
    # Append its corresponding number to Num_of_vehicles_1415
    for x in vehicle_1415:

        if x[0].upper() == state:
            num = x[1]
            break
    Num_of_vehicles_1415.append(num) 
    
# Create a dataframe containing shapes, state names and no. of vehicles   
df_poly = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in m.INDIA],
        'area': [area['STATE'] for area in m.INDIA_info],
        'Num_of_vehicles_1415' : Num_of_vehicles_1415
    })

'''
STEP 5 : USING DATA TO COLOR AREAS
'''
# Get all the shapes
shapes = [Polygon(np.array(shape), True) for shape in m.INDIA]
# Create a colormap
cmap = plt.get_cmap('Oranges')   
# Create a patch collection. Create patches on the top of the map, not beneath it (zorder=2)
pc = PatchCollection(shapes, zorder=2)

norm = Normalize()
# Set color according to the number of vehicle of the state
pc.set_facecolor(cmap(norm(df_poly['Num_of_vehicles_1415'].fillna(0).values)))
ax.add_collection(pc)

printed_names = []
for shapedict,state in zip(m.INDIA_info, m.INDIA):
    state_name = [value for key, value in shapedict.items()][0]
    
    if state_name in printed_names: continue
    # center of polygon
    x, y = np.array(state).mean(axis=0)
    # You have to align x,y manually to avoid overlapping for little states
    plt.text(x+.1, y, state_name, ha="center")
    printed_names += [state_name,] 


# Create a mapper to map color intensities to values
mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
mapper.set_array(Num_of_vehicles_1415)
plt.colorbar(mapper, shrink=0.4)
# Set title for the plot
ax.set_title("NUMBER OF VEHICLES OF INDIAN STATES FOR THE YEAR 2014-2015")
# Change plot size and font size
# plt.rcParams['figure.figsize'] = (20, 20)
# plt.rcParams.update({'font.size': 20})
plt.show()
df['increase %'] = (-(df['No. of Motor Vehicles/1000 population - During 2013-14'] - df['No. of Motor Vehicles/1000 population - During 2014-15'])/df['No. of Motor Vehicles/1000 population - During 2013-14'])*100
df.head()

change_1315 = []
for i in range(len(df)):
    states = df.iloc[i,0]
    percent_1315 = df.iloc[i,3]
    change_1315.append((states,percent_1315))
change_1315
'''
STEP 2 : CREATING A MAP
'''
# Create figure 
fig, ax = plt.subplots()
# Create a map with the coordinates determined by the Bounding Box tool
m = Basemap(projection='merc',lat_0=54.5, lon_0=-4.36,llcrnrlon=68.1, llcrnrlat= 6.5, urcrnrlon=97.4, urcrnrlat=35.5)
# Draw map boundary and set the color
m.drawmapboundary(fill_color='#46bcec')
# Fill continents and lakes
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
# Draw coast lines
m.drawcoastlines()

'''
STEP 3 : USING SHAPEFILES FOR DRAWING STATES 
'''
# Load the shape file of India
m.readshapefile(
    "../input/shapefile/india_shape/India_Shape/india_st","INDIA")


'''
STEP 4 : CREATING A DATAFRAME MAPPING SHAPES TO STATE NAME AND CHANGE IN NUMBER OF VEHICLES
'''
# Create an empty list to hold change in number of vehicles in 2014-15
change_of_vehicles_1315 = []

for state_info in m.INDIA_info:
    # Get the state in uppercase, as our csv file has state names in upper case
    state = state_info['STATE'].upper()
    # initialize number of vehicles = 0
    num = 0
    
    # In vehicle_1314 (containing tuples of state name and change in no. of vehicles), search for state 'state'
    # Append its corresponding number to change_of_vehicles_1315
    for x in change_1315:

        if x[0].upper() == state:
            num = x[1]
            break
    change_of_vehicles_1315.append(num) 
    
# Create a dataframe containing shapes, state names and change in no. of vehicles   
df_poly = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in m.INDIA],
        'area': [area['STATE'] for area in m.INDIA_info],
        'change_of_vehicles_1315' : change_of_vehicles_1315
    })

'''
STEP 5 : USING DATA TO COLOR AREAS
'''
# Get all the shapes
shapes = [Polygon(np.array(shape), True) for shape in m.INDIA]
# Create a colormap
cmap = plt.get_cmap('Oranges')   
# Create a patch collection. Create patches on the top of the map, not beneath it (zorder=2)
pc = PatchCollection(shapes, zorder=2)

norm = Normalize()
# Set color according to the change in number of vehicle of the state
pc.set_facecolor(cmap(norm(df_poly['change_of_vehicles_1315'].fillna(0).values)))
ax.add_collection(pc)

printed_names = []
for shapedict,state in zip(m.INDIA_info, m.INDIA):
    state_name = [value for key, value in shapedict.items()][0]
    
    if state_name in printed_names: continue
    # center of polygon
    x, y = np.array(state).mean(axis=0)
    # You have to align x,y manually to avoid overlapping for little states
    plt.text(x+.1, y, state_name, ha="center")
    printed_names += [state_name,] 


# Create a mapper to map color intensities to values
mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
mapper.set_array(change_of_vehicles_1315)
plt.colorbar(mapper, shrink=0.4)
# Set title for the plot
ax.set_title("CHANGE IN NUMBER OF VEHICLES OF INDIAN STATES PER 1000 POPULATION FOR THE YEAR 2013-2015")
# Change plot size and font size
# plt.rcParams['figure.figsize'] = (30, 30)
# plt.rcParams.update({'font.size': 15})
plt.show()
