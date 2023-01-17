import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

from mpl_toolkits.basemap import Basemap

from matplotlib import style



style.use('ggplot')

%matplotlib inline
data = pd.read_csv('../input/Mountains.csv')

data.head()
# lets check the missing values

data.info()
# fill in the missing values and assign some values to words.



data['Ascents bef. 2004'] = [170 if '>' in str(x) else (150 if 'Many' in str(x) else(0 if str(x)=='nan' or x is None or x=='' else x)) for x in data['Ascents bef. 2004']]

data['Ascents bef. 2004'] = data['Ascents bef. 2004'].astype(int)

data['Failed attempts bef. 2004'].fillna(0, inplace=True)

data['Failed attempts bef. 2004'] = data['Failed attempts bef. 2004'].astype(int)
# Check the top 20 failed attempts to ascent the peak.

# We see that the failed attempts are in relation with the successfull ascents on a peak i.e. failed attempts

# are higher where there are more successfull ascents on a peak which further tells about the concentration of

# attempts made by people on a peak.



fig, ax = plt.subplots(2,1, figsize=((8,8)))



ax = plt.subplot2grid((2,1), (1,0))

failed_df = data.sort_values('Failed attempts bef. 2004', ascending=False)

failed_df['Failed attempts bef. 2004'][:19][::-1].plot.barh()

ax.set_title('Top 20 Failed attempts')

ax.set_yticklabels(failed_df.Mountain[:19][::-1])



ax = plt.subplot2grid((2,1), (0,0))

success_df = data.sort_values('Ascents bef. 2004', ascending=False)

success_df['Ascents bef. 2004'][:19][::-1].plot.barh()

ax.set_title('Top 20 Successfull Ascents')

ax.set_yticklabels(failed_df.Mountain[:19][::-1])



plt.tight_layout()
# Lets check the total number of attempts to ascent made on a peak.



data['total_attempts'] = data['Failed attempts bef. 2004'] + data['Ascents bef. 2004']

data.sort_values('total_attempts', ascending=False)['total_attempts'][:19][::-1].plot.barh()

ax = plt.gca()

ax.set_yticklabels(data.sort_values('total_attempts', ascending=False)['Mountain'][:19][::-1])

ax.set_title('Top 20 Most Attempted Peaks')
# This is to check the relation of successfull and failed ascents on a peak against their 

# height in meters,

# their prominance in meters and rank in the world.



# The relation ship can be seen in the figure below.



fig, ax = plt.subplots(3,2, figsize=(10,8))



ax = plt.subplot2grid((3,2), (0,0))

ax.set_title('Height in meter vs Ascents')

ax.set_ylabel('Height')

ax.set_xlabel('Success')

ax.scatter(data['Ascents bef. 2004'], data['Height (m)'].astype(int), )



ax = plt.subplot2grid((3,2), (0,1))

ax.set_title('Height in meter vs Failed Attempts')

ax.set_ylabel('Height')

ax.set_xlabel('Failiure')

ax.scatter(data['Failed attempts bef. 2004'], data['Height (m)'].astype(int),)



ax = plt.subplot2grid((3,2), (1,0))

ax.set_title('Prominance in meter vs Ascents')

ax.set_ylabel('Prominance')

ax.set_label('Success')

ax.scatter(data['Ascents bef. 2004'], data['Prominence (m)'].astype(int), )



ax = plt.subplot2grid((3,2), (1,1))

ax.set_title('Prominance in meter vs Failed Attempts')

ax.set_ylabel('Prominance')

ax.set_xlabel('Failiure')

ax.scatter(data['Failed attempts bef. 2004'], data['Prominence (m)'].astype(int), )



ax = plt.subplot2grid((3,2), (2,0))

ax.set_title('Rank vs Ascents')

ax.set_ylabel('Rank')

ax.set_xlabel('Success')

ax.scatter(data['Ascents bef. 2004'], data.Rank)



ax = plt.subplot2grid((3,2), (2,1))

ax.set_title('Rank vs Failed Attempts')

ax.set_ylabel('Rank')

ax.set_xlabel('Failiure')

ax.scatter(data['Failed attempts bef. 2004'], data.Rank)



plt.tight_layout()

# convert the coordinates into decimal notation to plot them on the map



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
coo = data.Coordinates.apply(parse_dms)
data['lat'], data['lon'] = [lat for lat, _ in coo], [lon for _, lon in coo]
# All of moutains approximately lies on the same strip, The largest mountain ranges

# karakoram, Himaliyas,



plt.figure(num=None, figsize=(17, 15), dpi=80, facecolor='w', edgecolor='k')

src_map = Basemap(projection='gall', llcrnrlat=-90, urcrnrlat=90,\

            llcrnrlon=-180, urcrnrlon=180, resolution='l')



src_map.drawcoastlines()

src_map.drawcounties()

src_map.fillcontinents(color='gainsboro')

src_map.drawmapboundary(fill_color='steelblue')



# x, y = src_map(86.925278, 27.988056)

x, y = src_map(data['lon'].values, data['lat'].values)

src_map.plot(x, y, '^', color='green', markersize=15)