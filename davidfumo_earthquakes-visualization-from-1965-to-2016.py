import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap # useful for ploting maps

import seaborn as sns

sns.set(style="darkgrid")

% matplotlib inline





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/database.csv')
data.head()
data.shape
data.info()
data.isnull().sum()
# Location Source



# Display most common sources



data['Location Source'].value_counts()[:5]
# plot Magnitude Source Frequency

plt.figure(figsize=(13, 10))



sns.countplot(x="Location Source", data=data)

# plt.ylabel('Frequency')

plt.title('Location Source and Frequency')
# Magnitude Source



# Display most common magnitude sources



data['Magnitude Source'].value_counts()[:5]
plt.figure(figsize=(13, 10))



sns.countplot(x="Magnitude Source", data=data)



plt.title('Magnitude Source and Frequency')
# Minumum magnitude



data['Magnitude'].min()
# Maximum magnitude



data['Magnitude'].max()
g8 = data[data['Magnitude'] > 8.5]

g8['Location Source'].value_counts()
# Plot Distribution plot of 'Magnitude' values



plt.hist(data['Magnitude'])



plt.xlabel('Magnitude Size')

plt.ylabel('Number of Occurrences')
# Plot Distribution plot of 'Magnitude Type' values



# plt.hist(data['Magnitude Type'])



# plt.figure(figsize=(13, 10))



sns.countplot(x="Magnitude Type", data=data)

plt.ylabel('Frequency')

plt.title('Magnitude Type VS Frequency')
# Latitude vs Longitude



# Simple distribution  maping



plt.figure(figsize=(13, 10))



sns.lmplot('Longitude', 'Latitude',

           data=data,

           fit_reg=False,

           scatter_kws={"marker": "D",

                        "s": 50})
def get_marker_color(magnitude):

    # Returns green for small earthquakes, yellow for moderate

    #  earthquakes, and red for significant earthquakes (This is just my assumption, it does not reflect

    # the real metric for small, moderate and significant earthquake)

    if magnitude < 6.2:

        return ('go')

    elif magnitude < 7.5:

        return ('yo')

    else:

        return ('ro')



# Make this plot larger.

plt.figure(figsize=(14,10))



eq_map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0,

              lat_0=0, lon_0=-130)

eq_map.drawcoastlines()

eq_map.drawcountries()

eq_map.fillcontinents(color = 'gray')

eq_map.drawmapboundary()

eq_map.drawmeridians(np.arange(0, 360, 30))

eq_map.drawparallels(np.arange(-90, 90, 30))

 

# read longitude, latitude and magnitude

lons = data['Longitude'].values

lats = data['Latitude'].values

magnitudes = data['Magnitude'].values

timestrings = data['Date'].tolist()

    

min_marker_size = 0.5

for lon, lat, mag in zip(lons, lats, magnitudes):

    x,y = eq_map(lon, lat)

    msize = mag # * min_marker_size

    marker_string = get_marker_color(mag)

    eq_map.plot(x, y, marker_string, markersize=msize)

    

title_string = "Earthquakes of Magnitude 5.5 or Greater\n"

title_string += "%s - %s" % (timestrings[0][:10], timestrings[-1][:10])

plt.title(title_string)



plt.show()
import datetime



data['date'] = data['Date'].apply(lambda x: pd.to_datetime(x))
# Earthquakes by Year



# Process the year from 'Date' column



data['year'] = data['date'].apply(lambda x: str(x).split('-')[0])
# Earthquakes by Year



plt.figure(figsize=(15, 8))

sns.set(font_scale=1.0)

sns.countplot(x="year", data=data)

plt.ylabel('Number Of Earthquakes')

plt.title('Number of Earthquakes In Each Year')
data['year'].value_counts()[:1]
### this gives wrong results



# Earthquakes Variations over the years

#plt.figure(figsize=(10, 8))



#x = data['year'].unique()

#y = data['year'].value_counts()



#plt.scatter(x, y)

#plt.xlabel('Year')

#plt.ylabel('Number of Earthquakes')

#plt.title('Earthquakes from 1995 to 2016')

#plt.show()
# Correct match (year, number of earthquakes)



x = data['year'].unique()

y = data['year'].value_counts()



count = []

for i in range(len(x)):

    key = x[i]

    count.append(y[key])



# Earthquakes Variations over the years



plt.figure(figsize=(10, 8))



plt.scatter(x, count)

plt.xlabel('Year')

plt.ylabel('Number of Earthquakes')

plt.title('Earthquakes Per year from 1995 to 2016')

plt.show()
data.loc[data['Magnitude'] > 8, 'Class'] = 'Great'

data.loc[ (data['Magnitude'] >= 7) & (data['Magnitude'] < 7.9), 'Class'] = 'Major'

data.loc[ (data['Magnitude'] >= 6) & (data['Magnitude'] < 6.9), 'Class'] = 'Strong'

data.loc[ (data['Magnitude'] >= 5.5) & (data['Magnitude'] < 5.9), 'Class'] = 'Moderate'
# Magnitude Class distribution



sns.countplot(x="Class", data=data)

plt.ylabel('Frequency')

plt.title('Magnitude Class VS Frequency')