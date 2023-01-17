import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
volcanoes = pd.read_csv("../input/volcanic-eruptions/database.csv")

#List of Columns on Valcano dataset
volcanoes.columns
#list of first 5 rows
volcanoes.head()
eq = pd.read_csv("../input/earthquake-database/database.csv")
eq.columns
#list of first 5 rows
eq.head()
#Changing the Column name just for easier handling purpose (ofcourse its not mandatory!)
volcanoes = volcanoes.rename(columns={'Elevation (Meters)': 'Elevation'})
volcanoes.head(2)
#Check the types of valcanoes thats existing
volcanoes['Type'].value_counts() 
def cleanup_type(s):
    if not isinstance(s, str):
        return s
    s = s.replace('?', '').replace('  ', ' ')
    s = s.replace('Stratovolcano(es)', 'Stratovolcano')
    s = s.replace('Shield(s)', 'Shield')
    s = s.replace('Submarine(es)', 'Submarine')
    s = s.replace('Pyroclastic cone(s)', 'Pyroclastic cone')
    s = s.replace('Volcanic field(s)', 'Volcanic field')
    s = s.replace('Caldera(s)', 'Caldera')
    s = s.replace('Complex(es)', 'Complex')
    s = s.replace('Lava dome(s)', 'Lava dome')
    s = s.replace('Maar(s)', 'Maar')
    s = s.replace('Tuff cone(s)', 'Tuff cone')
    s = s.replace('Tuff ring(s)', 'Tuff ring')
    s = s.replace('Fissure vent(s)', 'Fissure vent')
    s = s.replace('Lava cone(s)', 'Lava cone')
    return s.strip().title()

volcanoes['Type'] = volcanoes['Type'].map(cleanup_type)
volcanoes['Type'].value_counts() 
#Now, lets check for any null values present
volcanoes.isnull().sum()
volcanoes.dropna(inplace=True)
len(volcanoes)
# A general check on why we are removing submarine alone..
volcanoes[volcanoes['Type'] == 'Submarine'].head()
volcanoes = volcanoes[volcanoes['Elevation'] >= 0]
len(volcanoes)
def plot_map(Longitude , Latitude, Elevation, projection='mill', llcrnrlat=-80, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='i', min_marker_size=2):
    bins = np.linspace(0, Elevation.max(), 10)
    marker_sizes = np.digitize(Elevation, bins) + min_marker_size

    m = Basemap(projection=projection, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution=resolution)
    m.drawcoastlines()
    m.drawmapboundary()
    m.fillcontinents(color = '#333333')

    for lon, lat, msize in zip(Longitude , Latitude, marker_sizes):
        x, y = m(lon, lat)
        m.plot(x, y, '^r', markersize=msize, alpha=.7)

    return m
from mpl_toolkits.basemap import Basemap
plt.figure(figsize=(16, 8))
plot_map(volcanoes['Longitude'], volcanoes['Latitude'], volcanoes['Elevation'])
plt.title('Volcanoes of the World', color='#000000', fontsize=20)
from mpl_toolkits.basemap import Basemap
plt.figure(figsize=(16, 8))
Vol = volcanoes[volcanoes['Type'] == 'Stratovolcano']
plot_map(Vol['Longitude'], Vol['Latitude'], Vol['Elevation'])
#List of Stratovalcanoes present in US.
Vol_US = volcanoes.loc[(volcanoes['Country'] == 'United States') & (volcanoes['Type'] == 'Stratovolcano')]
Vol_US.head()
len(Vol_US)
# Finding the highest peak among the list
Vol_US['Elevation'].max()
Vol_US.loc[Vol_US['Elevation'] == 5005]
# Finding the lowest peak among the list
Vol_US['Elevation'].min()
Vol_US.loc[Vol_US['Elevation'] == 0]
# Displaying the two particular rows of min and Max values
Vol_US = Vol_US.loc[(675,967),:]
Vol_US
plt.figure(figsize=(12, 10))
plot_map(Vol_US['Longitude'], Vol_US['Latitude'],Vol_US['Elevation'],min_marker_size=10)
plt.figure(figsize=(12, 10))
plot_map(volcanoes['Longitude'], volcanoes['Latitude'],volcanoes['Elevation'],
         llcrnrlat=5.5, urcrnrlat=83.2, llcrnrlon=-180, urcrnrlon=-52.3, min_marker_size=4)
m = Basemap(projection='mill',llcrnrlat=5.5, urcrnrlat=83.2, llcrnrlon=-180, urcrnrlon=-52.3,resolution='c')
fig = plt.figure(figsize=(12,10))

longitudes_vol = volcanoes['Longitude'].tolist()
latitudes_vol = volcanoes['Latitude'].tolist()

longitudes_eq = eq['Longitude'].tolist()
latitudes_eq = eq['Latitude'].tolist()

x,y = m(longitudes_vol,latitudes_vol)
a,b= m(longitudes_eq,latitudes_eq)

plt.title("Volcanos areas (red) Earthquakes (green)", color='#000000', fontsize=20)
m.plot(x, y, '^r', markersize = 5, color = 'red')
m.plot(a, b, "o", markersize = 3, color = 'green')

m.drawcoastlines()
m.drawcountries()
#m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()
plt.figure(figsize=(12, 10))
plot_map(volcanoes['Longitude'], volcanoes['Latitude'], volcanoes['Elevation'],
         llcrnrlat=-57, urcrnrlat=15, llcrnrlon=-87, urcrnrlon=-32, min_marker_size=4)
plt.figure(figsize=(18, 8))
plot_map(volcanoes['Longitude'], volcanoes['Latitude'], volcanoes['Elevation'],
         llcrnrlat=-11.1, urcrnrlat=6.1, llcrnrlon=95, urcrnrlon=141.1, min_marker_size=4)
m1 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
fig = plt.figure(figsize=(12,10))

longitudes_eq = eq['Longitude'].tolist()
latitudes_eq = eq['Latitude'].tolist()
k,j= m1(longitudes_eq,latitudes_eq)

plt.title("Earthquake of the World", color='#000000', fontsize=20)
m1.plot(k, j, "o", markersize = 3, color = 'green')

m1.drawcoastlines()
m1.drawcountries()
m1.drawmapboundary()
plt.show()
#Selecting the necessary columns for Analysis
eq_analysis = eq[["Date", "Time", "Latitude","Longitude","Magnitude", "Depth"]]
len(eq_analysis)
eq_analysis.head()
m2 = Basemap(projection='mill')
rof_lat = [-61.270, 56.632]
rof_long = [-70, 120]
ringoffire = eq_analysis[((eq_analysis.Latitude < rof_lat[1]) & 
                    (eq_analysis.Latitude > rof_lat[0]) & 
                     ~((eq_analysis.Longitude < rof_long[1]) & 
                       (eq_analysis.Longitude > rof_long[0])))]
f,g = m2([longs for longs in ringoffire["Longitude"]],
         [lats for lats in ringoffire["Latitude"]])
fig3 = plt.figure(figsize=(20,20))
plt.title("Earthquakes in the Ring of Fire Area")
m2.plot(f,g, "o", markersize = 3, color = 'green')
m2.drawcoastlines()
m2.drawmapboundary()
m2.drawcountries()
m2.fillcontinents(color='lightsteelblue',lake_color='skyblue')

plt.show()
print("Total number of data on world's Earthquakes:", len(eq_analysis))
print("Total number of Earthquakes in the Ring of Fire Area:",len(ringoffire))
eq_max = eq_analysis['Magnitude'].max()
eq_min = eq_analysis['Magnitude'].min()
print (eq_max)
print (eq_min)
max_eq = eq_analysis.loc[eq_analysis['Magnitude'] == 9.1] 
len(max_eq)
min_eq = eq_analysis.loc[eq_analysis['Magnitude'] == 5.5]
len(min_eq)
#Scroll in find the two most horrible earthquakes being hit.
import folium
map = folium.Map(location = [eq_analysis['Latitude'].mean(), eq_analysis['Longitude'].mean()], zoom_start = 4, tiles = 'Mapbox Bright' )
folium.Marker(
    location=[3.295, 95.982],
    popup='Indonesia',
    icon=folium.Icon(icon='cloud')
).add_to(map)
folium.Marker(
    location=[38.297, 142.373],
    popup='Japan',
    icon=folium.Icon(color='green')
).add_to(map)

map
eq_Japan = eq[eq['Location Source'] == 'Japan'] 
eq_Japan_list = eq.loc[:,('Latitude','Longitude')]
Vol_Japan = volcanoes.loc[volcanoes['Country'] == 'Japan'] 
Vol_Japan_list = Vol_Japan.loc[:,('Latitude','Longitude')]
len(Vol_Japan)
plt.figure(figsize=(18, 8))
plot_map(Vol_Japan['Longitude'], Vol_Japan['Latitude'], Vol_Japan['Elevation'],
         llcrnrlat=25, urcrnrlat=46, llcrnrlon=125, urcrnrlon=150, min_marker_size=4)
n = pd.read_csv("../input/earthquake-database/database.csv",encoding='ISO-8859-1')
n=n[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
n.head()
import time
#getting the year from Date Attribute
n['Year']= n['Date'].str[6:]

fig = plt.figure(figsize=(10, 10))
fig.text(.8, .3, 'L.Rajesh', ha='right')
cmap = plt.get_cmap('coolwarm')

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 10)
m.drawmapboundary(fill_color='lightblue')


START_YEAR = 1965
LAST_YEAR = 2016

points = n[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][n['Year']==str(START_YEAR)]

x, y= m(list(points['Longitude']), list(points['Latitude']))
scat = m.scatter(x, y, s = points['Magnitude']*points['Depth']*0.3, marker='o', alpha=0.3, zorder=10, cmap = cmap)
year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)
plt.title("Earthquake visualisation (1965 - 2016)")
plt.close()

start = time.time()
def update(frame_number):
    current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))
    year_text.set_text(str(current_year))
    points = n[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][n['Year']==str(current_year)]
    x, y= m(list(points['Longitude']), list(points['Latitude']))
    color = points['Depth']*points['Magnitude'];
    scat.set_offsets(np.dstack((x, y)))
    scat.set_sizes(points['Magnitude']*points['Depth']*0.3)

ani = animation.FuncAnimation(fig, update, interval=500, repeat_delay=0, frames=LAST_YEAR - START_YEAR + 1,blit=False)
ani.save('animation.gif', writer='imagemagick') #, writer='imagemagick'
plt.show()

end = time.time()
print("Time taken by above cell is {}".format(end-start))
    
#ani = animation.FuncAnimation(fig, update, interval=750, frames=LAST_YEAR - START_YEAR + 1)
#ani.save('animation.gif', writer='imagemagick', fps=5)
import io
import base64

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
