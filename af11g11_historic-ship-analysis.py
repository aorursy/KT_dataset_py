# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as datetime
# For rendering animations in notebook:
from IPython.display import HTML
import io
import base64
# For me
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Look at all of the columns that exist in dataset
df = pd.read_csv('../input/CLIWOC15.csv')
for clm in df.columns:
    print(clm)
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap  

fig = plt.figure(figsize=(20,10))

# Setup Base Map
m = Basemap(projection='robin',lon_0=180,resolution='c', llcrnrlon=120, urcrnrlon=-30)
m.drawcoastlines()
m.drawcountries()
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))
m.fillcontinents(color='grey')

# Lets try trace one ships journey across the globe
ship_name = df['ShipName'].unique()[342]
print('Ship name = ', ship_name)

# Make a dataframe for a specific ship
df_ship = df[df['ShipName'] == ship_name]
df_ship = df_ship.reset_index()

# Have a sneaky look at where it travelled to and from
departure_port = df_ship['VoyageFrom'].unique()
arrival_port = df_ship['VoyageTo'].unique()

print('Journey: ', departure_port , '-', arrival_port)

# Function to create a numpy array given a dataframe and dataframe index (column)
def get_numpy_array(df, string):
    output = df[string]
    output = output[~np.isnan(output)]
    return output

# Get array values for those with one axis
year = get_numpy_array(df_ship, 'Year')
month = get_numpy_array(df_ship, 'Month')
day = get_numpy_array(df_ship, 'Day')

# Create an matrix with x, y coordinates 
lat = df_ship['Lat3']
lon = df_ship['Lon3']
coord = np.column_stack((list(lon),list(lat)))
# Remove nan values
coord = coord[~np.isnan(coord).any(axis=1)]

# Get the x and y coordinates 
x, y = m(coord[:,0], coord[:,1])
# Draw the path on the map (reduce alpha so that it is faded slightly)
m.plot(x, y,'.', color='grey', alpha=0.9)

# Lets set up the animation
x,y = m(0, 0)
# Point will be altered by the init and animate functions
# Essentially moving the points on the plot
point = m.plot(x, y, 'o', markersize=7, color='red')[0]

# Set up the animation (empty points)
def init():
    point.set_data([], [])
    return point,

# Function for animating the data; takes coordinates at time equal to i
def animate(i):
    # Transform the coordinates
    x, y = m(coord[i,0], coord[i,1])
    point.set_data(x,y)
    # Set the date as the title (English format)
    fig.suptitle('%2i / %2i / %4i \n %s \n %s - %s' % (day[i], month[i], year[i], 
                                                    ship_name,
                                                    departure_port, arrival_port))
    return point,

# Create the output
output = animation.FuncAnimation(fig,
                                 animate,
                                 init_func=init,
                                 frames=len(coord),
                                 interval=100,
                                 blit=True,
                                 repeat=False)
# Write the output
# output.save('ship_journey.gif', writer='imagemagick')
# Show last frame on screen
# plt.show()

# ani = animation.FuncAnimation(fig,animate,sorted(df.pickup_hour.unique()), interval = 1000)
plt.close()
output.save('ship_journey.gif', writer='imagemagick', fps=2)
filename = 'ship_journey.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
# Key for ship colours:
# Try to match up with national colours at time - TODO
national_color_dict = {'Spanish'  : 'yellow',
                       'British'  : 'red',
                       'British ' : 'red',
                       'French'   : 'blue',
                       'Dutch'    : 'orange',
                       'Hamburg'  : 'green',
                       'American' : 'navy',
                       'Danish'   : 'pink',
                       'Swedish'  : 'purple'}
# Make a new column called YearDay will allow iteration through each day 
# removes the hour etc columns that could cause difficulty when plotting all ships present
# df['YearDay'] = df['Year'].map(str) + df['Day'].map(str)
df['YearMonthDay'] = df['UTC'].astype(str).str[:-2].astype(np.int64)
# Sort the values (so animation is in order)
df = df.sort_values(by=['YearMonthDay'], ascending=True)
# Reset index
df = df.reset_index()
# Have a look at the head
print("\n YearMonthDay head = ", df['YearMonthDay'].head(5))
# How many values
print("\n Number of YearMonthDay = ", df['YearMonthDay'].count())

df['YearMonthDay'][124]
# Try and look at all ships present at a certain date

# First lets try plot all ships present at a certain date

# Take a temporary array at a certain YearDay
df_temp = df[df['YearMonthDay']==df['YearMonthDay'][10000]]
print("\n Chosen YearMonthDay = ", df_temp['YearMonthDay'].unique()[0])

# Will still have multiple entries 
# Will take last entry from the ships log for the day
df_temp = df_temp.drop_duplicates(subset='ShipName', keep='last')

# Generate base map
plt.figure(figsize=(20,10))
# Setup Base Map
m = Basemap(projection='robin',lon_0=180,resolution='c', llcrnrlon=120, urcrnrlon=-30)
m.drawcoastlines()
m.drawcountries()
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))
m.fillcontinents(color='grey')

# Now would like to plot position of each ship
for index, row in df_temp.iterrows():
    # Add points to world map
    m.scatter(row['Lon3'], row['Lat3'],
              color=national_color_dict[row['Nationality']],
              marker='o', latlon=True)
# Set title
plt.title('%2i / %2i / %4i' % (df_temp['Day'].unique()[0],
                               df_temp['Month'].unique()[0],
                               df_temp['Year'].unique()[0]))
plt.show()
# Animate ship positions
# Generate base map
fig = plt.figure(figsize=(20,10))
# Setup Base Map
m = Basemap(projection='robin',lon_0=180,resolution='c', llcrnrlon=120, urcrnrlon=-30)
m.drawcoastlines()
m.drawcountries()
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))
m.fillcontinents(color='grey')

#####################################

# Lets animate the path
national_colours = 'British'
scat = m.scatter([0], [0],
                 color=national_color_dict[national_colours],
                 marker='o',
                 latlon=True)

def init():
    return scat,

# Function for animating the data; takes coordinates at time equal to i
def update_plot(yearmonthday):
#     Make temprary dataframe for passed in year day
    df_temp = df[df['YearMonthDay']==yearmonthday]
    
    # Will still have multiple entries 
    # Will take last entry from the ships log for the day
    df_temp = df_temp.drop_duplicates(subset='ShipName', keep='last')
    
    # Drop NA lon and lat
    # df_temp = df_temp.dropna(subset=['Lon3', 'Lat3'])

    # Get x, y arrays
    longitude_array = df_temp['Lon3'].values
    latitude_array = df_temp['Lat3'].values
    
    # Transform coordinates
    x, y = m(longitude_array, latitude_array)
    # Arange coordinates for correct format for passing to set offsets
    coordinates = np.vstack([x, y]).T
    
    # Get national colour array
    national_colours = df_temp['Nationality']
    df_temp['NationalColor'] = df_temp['Nationality'].map(national_color_dict) 
    
    # Pass arrays of coordinates and colours to scat
    scat.set_offsets(coordinates)
    scat.set_color(df_temp['NationalColor'])
    
    # Set the date as the title (English format)
    fig.suptitle('%2i / %2i / %4i' % (df_temp['Day'].values[0],
                                      df_temp['Month'].values[0],
                                      df_temp['Year'].values[0]))
     
    print('\r', 'YearMonthDay', yearmonthday, ' / ', df['YearMonthDay'].nunique(), end='')

    return scat,

# Prep dataframe again
# Just take a certain Year?
df = df[df['Year'].between(1774,1775)]

# Create the output
output = animation.FuncAnimation(fig,                                                #
                                 update_plot,                                              #
                                 init_func=init,                                           #
                                 frames=np.sort(df['YearMonthDay'].unique())[:],           #
                                 interval=100,                                             # Interval between frames in ms
                                 blit=False,                                               # TODO: what is blit?
                                 repeat=False)                                             #
# # Write the output
# output.save('European_Historical_Ship_Travels.gif', writer='imagemagick')
# # Show last frame on screen
# plt.show()
plt.close()
output.save('European_Historical_Ship_Travels.gif', writer='imagemagick', fps=2)
filename = 'European_Historical_Ship_Travels.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
i = 0
for i in range(10):
    print('\r', 'Iteration', i, 'Score:', 3, end='')
    time.sleep(1)
#

df2 = df[df['Year'].between(1774,1784)]
df2['Year'].unique()