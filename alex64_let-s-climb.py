import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import style

import re



style.use('ggplot')

%matplotlib inline



data = pd.read_csv('../input/Mountains.csv')

data.head()
data.info()
data.drop(data.columns[3], axis=1, inplace=True)

data.rename(columns={'Height (m)': 'Height', 'Prominence (m)': 'Prominence',

                   'Ascents bef. 2004' : 'Success', 'Failed attempts bef. 2004' : 'Failed' }, inplace=True)

data['Failed'] = data['Failed'].fillna(0).astype(int)

data['Success'] = [200 if '>' in str(x) else (160 if 'Many' in str(x) else x) for x in data['Success']]

data['Success'] = data['Success'].fillna(0).astype(int)

data['First ascent'] = [2020 if 'unclimbed' in str(x) else x for x in data['First ascent']]

data['First ascent'] = data['First ascent'].fillna(0).astype(int)
plt.hist(data['First ascent'], bins = 22)

plt.ylabel('# mountains')

plt.xlabel('year')

plt.title('First ascents')
data['Height'].plot.hist(color = 'steelblue', bins = 20)

plt.bar(data['Height'], (data['Height']-data['Height'].min())/(data['Height'].max()-data['Height'].min())*23, color = 'orange', 

       width = 30, alpha = 0.2)

plt.ylabel('# mountains')

plt.xlabel('Height')

plt.text(8750,20, "Height", color = 'orange')

plt.title('Height of mountains')
data['Range0'] = [ 'Himalaya' if 'Himalaya' in str(x) else ('Karakoram' if 'Karakoram' in str(x) \

                                                           else ('Pamir' if 'Pamir' in str(x) else 'Other')) 

                 for x in data['Range'] ]
fig = plt.figure(figsize=(11, 6))

fig.add_subplot(332)

dH = data.groupby('Range0').size()

dH.name = ''

dH.plot.pie(shadow = True)

plt.title('Count of High Mountains')



fig.subplots_adjust(wspace=0.9, hspace = -0.4)

ax1 = fig.add_subplot(223)

dH = data.groupby('Range0').get_group('Himalaya').groupby('Range').size()

dH.name = ''

dH.plot.pie(ax = ax1, shadow = True, title = 'Himalaya')

dH = data.groupby('Range0').get_group('Himalaya').groupby('Range').size()

dH.name = ''

ax2 = fig.add_subplot(224)

dH.plot.pie(ax = ax2, shadow = True, title = 'Karakoram')
data['Attempts'] = data['Failed'] + data['Success']
fig = plt.figure(figsize=(13, 7))

fig.add_subplot(211)

plt.scatter( data['First ascent'], data['Height'], c = data['Attempts'], alpha = 0.8, s = 50)

plt.ylabel('Height')

plt.xlabel('First ascent')



fig.add_subplot(212)

plt.scatter( data['First ascent'], data['Rank'].max() - data['Rank'], c = data['Attempts'], alpha = 0.8, s = 50)

plt.ylabel('Rank')

plt.xlabel('First ascent')
data0 = data.copy()

data = data[data['First ascent'] <= 2004]

Everest = data[data['Height'] == 8848]

data = data[data['Height'] != 8848]

data = data[data["Attempts"] != 0]
fig = plt.figure(figsize=(13, 13))

fig.add_subplot(211)

plt.scatter( data['Height'], data['Prominence'], s = data['Attempts']*5, c = data['First ascent'], alpha = 0.8 )

plt.ylabel('Height')

plt.xlabel('Prominance')
data['Difficulty'] = data['Attempts']/data['Success']

data['Difficulty'] = (data['Difficulty'])/(data['Difficulty'].max())

Everest['Difficulty'] = (Everest['Attempts']/Everest['Success'])/(data['Difficulty'].max())

df = data[data['Attempts']>4].sort_values(by = 'Difficulty', ascending = True)
plt.figure(figsize = (12, 8))

plt.subplot(231)

plt.scatter( data['First ascent'], data['Height'], alpha = 0.8, s = 50)

plt.ylabel('Height')

plt.xlabel('First ascent')



plt.subplot(232)

plt.scatter( data['Attempts'], data['Prominence'], alpha = 0.8, s = 50)

plt.ylabel('Prominence')

plt.xlabel('Attempts')



plt.subplot(233)

plt.scatter( data['Attempts'], data['Height'], alpha = 0.8, s = 50)

plt.ylabel('Height')

plt.xlabel('Attempts')



plt.subplot(234)

plt.scatter( data['Height'], data['Difficulty'], alpha = 0.8, s = 50)

plt.ylabel('Difficulty')

plt.xlabel('Height')



plt.subplot(235)

plt.scatter( data['First ascent'], data['Difficulty'], alpha = 0.8, s = 50)

plt.ylabel('Difficulty')

plt.xlabel('First ascent')

plt.plot()
data[data.Attempts > 150]
print(df[["Mountain", "Height", "Attempts", "Success"]][-10:])

df[-10:].plot.barh(x='Mountain', y='Difficulty')
bigData = data[data['Height'] >= 8000].append(Everest).sort_values('Attempts', ascending=False)

print(bigData[["Mountain", "Height", "Attempts", "Success", "Difficulty"]][:10])

bigData['Attempts'] = bigData['Attempts']/bigData['Attempts'].max()

bigData['Difficulty'] = bigData['Difficulty']/bigData['Difficulty'].max()



bigData[:10][::-1].plot.barh(x='Mountain', y=['Attempts', 'Difficulty'], stacked=False)
bigData = bigData[bigData['Height'] != 8848]

bigData[:10][::-1].plot.barh(x='Mountain', y=['Attempts', 'Difficulty'], stacked=False)
# convert the coordinates into decimal notation to plot them on the map

# Here I use functions written by SalmanGhauri.



def dms2dd(degrees, minutes, seconds, direction):

    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);

    if direction == 'S' or direction == 'W':

        dd *= -1

    return dd



def parse_dms(dms):

    parts = re.split('[^\d\w]+', dms)

    lat = dms2dd(parts[0], parts[1], parts[2], parts[3])

    lng = dms2dd(parts[4], parts[5], parts[6], parts[7])

    return lat, lng
coo = data0.Coordinates.apply(parse_dms)

data0['lat'], data0['lon'] = [lat for lat, _ in coo], [lon for _, lon in coo]

lons = data0['lon'].values

lats =  data0['lat'].values

colors = data0['Range0'].map( {'Himalaya': 'green', 'Karakoram': 'red', 'Pamir': 'black', 'Other': 'yellow'} )#.astype(int) 

heights = data0['Height'].values
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.basemap import Basemap

from matplotlib.collections import PolyCollection



with plt.xkcd():

    plt.figure(figsize=([11, 8]))

    m_g = Basemap(resolution='i', projection='robin', lon_0=0)

    x, y = m_g(lons, lats)

    m_g.drawlsmask(land_color='.8', ocean_color='#f7fbff')

    m_g.drawcoastlines(color='.8', linewidth=1)

    m_g.scatter(x, y, color='blue')

plt.show()
map = Basemap(llcrnrlon=25,llcrnrlat=-20,urcrnrlon=160,urcrnrlat=90,)



fig = plt.figure(figsize = [13, 6])



ax1 = fig.add_subplot(1, 2, 1, projection='3d')

ax1.set_axis_off()

ax1.azim = 300

ax1.dist = 5.5



polys = []

for polygon in map.landpolygons:

    polys.append(polygon.get_coords())





lc = PolyCollection(polys, edgecolor='black',

                    facecolor='steelblue', closed=True)



ax1.add_collection3d(lc)

ax1.add_collection3d(map.drawcoastlines(linewidth=0.25))

ax1.add_collection3d(map.drawcountries(linewidth=0.35, color = 'black'))

x, y = map(lons, lats)

ax1.bar3d(x, y, np.zeros(len(x)), 2, 2, (heights-heights.min()), color= colors, alpha=0.7)



ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax2.set_axis_off()

ax2.azim = 15

ax2.dist = 5.5

map2 = Basemap(llcrnrlon=45,llcrnrlat=-10,urcrnrlon=150,urcrnrlat=60,)

polys = []

for polygon in map2.landpolygons:

    polys.append(polygon.get_coords())

lc = PolyCollection(polys, edgecolor='black',

                    facecolor='steelblue', closed=True)

ax2.add_collection3d(lc)

ax2.add_collection3d(map2.drawcoastlines(linewidth=0.25))

ax2.add_collection3d(map2.drawcountries(linewidth=0.35, color = 'black'))

x, y = map2(lons, lats)

ax2.bar3d(x, y, np.zeros(len(x)), 2, 2, (heights-heights.min()), color= colors, alpha=0.7)



ax2.text2D(1.2, 0.85, 'Himalaya', bbox={'facecolor':'green', 'alpha':0.5, 'pad':10}, transform=ax2.transAxes)

ax2.text2D(1.2, 0.65, "Karakoram", bbox={'facecolor':'red', 'alpha':0.5, 'pad':10}, transform=ax2.transAxes)

ax2.text2D(1.2, 0.45, 'Pamir', bbox={'facecolor':'black', 'alpha':0.5, 'pad':10}, transform=ax2.transAxes)

ax2.text2D(1.2, 0.25, "Other", bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10}, transform=ax2.transAxes)

ax2.text2D(1.2, 0.05, "Height of mountains", transform=ax2.transAxes)



plt.show()