import gpxpy
import glob #for listing and opening every file un folder
import os
import pandas as pd
os.chdir('..//input//')

#distace between 2 points function
import math
def dist(lat1,lon1,lat2,lon2):
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d
path = '..//input//'
names42k = [os.path.splitext(filename)[0] for filename in os.listdir(path)]
marathons=[]
columns = ['Longitude', 'Latitude', 'Altitude']
for filename in glob.glob('*.gpx'):
   
    #parsing and creating df
    gpx = gpxpy.parse(open(filename,encoding='utf-8'))
    track = gpx.tracks[0]
    segment = track.segments[0]

    data = []
    segment_length = segment.length_3d()
    for point in segment.points:
        data.append([point.longitude, point.latitude,point.elevation])
    marathons.append(pd.DataFrame(data, columns=columns))
    
for j in range (len(marathons)):
    
    
    marathons[j]['Distance']=0

    #addind 'distance passed' column
    for i in range(0,len(marathons[j].index)-1):
        lat1=marathons[j]['Latitude'].iloc[i]
        lon1=marathons[j]['Longitude'].iloc[i]
        lat2=marathons[j]['Latitude'].iloc[i+1]
        lon2=marathons[j]['Longitude'].iloc[i+1]
        marathons[j]['Distance'].iloc[i+1]=dist(lat1,lon1,lat2,lon2)
    #adding cumulative distance
    marathons[j]['cumulative'] = marathons[j]['Distance'].cumsum(axis = 0)

for marathon in marathons:
    marathon.dropna(inplace=True)

import matplotlib.pyplot as plt
import pylab as py
plt.style.use('fivethirtyeight')
fig=plt.figure(figsize=(18,10))
for i in range(6):
    plt.plot(marathons[i]['cumulative'], marathons[i]['Altitude'],label=names42k[i])
plt.legend(prop={'size': 18})
plt.title('Marathon height profiles', size=20)
plt.ylabel('meters',size=16)
plt.xlabel('km', size=16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.tight_layout()

#normlising height so that all marathons start at 0
for marathon in marathons:
    marathon['normilised height']=marathon['Altitude']-marathon['Altitude'][2]
fig=plt.figure(figsize=(18,7))
for i in range(6):
    plt.plot(marathons[i]['cumulative'], marathons[i]['normilised height'],label=names42k[i])
    #py.fill_between(marathons[i]['cumulative'], marathons[i]['Altitude'],0, alpha=0.3)
plt.legend(prop={'size': 18})
plt.title('Normalized marathon height profile', size=20)
plt.ylabel('meters',size=16)
plt.xlabel('km', size=16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.tight_layout()

plt.style.use('fivethirtyeight')
#total climb and descent
x = ['Bible','Eiltat','Jerusalem','Sovev emek','Tiberias','TLV']
negative_data = [-454,-615,-669,-680,-191,-130]
positive_data = [1097,615,659,679,191,130]

fig = plt.figure(figsize=(18,7) )
ax = plt.subplot(111)
ax.bar(x, negative_data)
ax.bar(x, positive_data)
ax.set_ylabel('meters')
ax.set_title('42K cumulative ascent and descent')
plt.tight_layout()

