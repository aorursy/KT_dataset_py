# Calculate vehicle stop locations

# 

# Parameters:

# - Min speed to assume vehicle stopped: 5km/h (to accomodate for GPS innacuracy)

# - Min time to record stop: 10 min





minSpeed=5

minStopDuration=10

inputData="/kaggle/input/taximovementconcatenated/allCars.csv"



print("Extracting stops longer than %s min from %s" % (minStopDuration,inputData))



# Todo:

#  - Rewrite calcStops to use dataframe

#  - Change from haversine to geopy.distance (as in cluster function)

#  - Adjust to accomodate for datasets with varying sampling frequency

# - Remove cars with too few samples (less than 500?)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import radians, cos, sin, asin, sqrt





# global allStops



# Used to calculate distance between coordinates

def haversine(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    # Radius of earth in kilometers is 6371

    km = 6371* c

    return km



# Generate list of all stops in file. Assumes constant sampletime

def calcStops(f,maxSpeedForStop, minTimeForStop):

    pos=pd.read_csv(f)

    pos=pos.to_numpy()

    i=0;

    alreadyStandingStill=False;

    firstStop=""

    minutesStandingStill=0

    stops=[]

    for i in range(1,len(pos)):

        distanceMoved=haversine(pos[i][1],pos[i][2],pos[i-1][1],pos[i-1][2])

        speed=distanceMoved*60

        if(speed < maxSpeedForStop and not alreadyStandingStill):

            firstStop=pos[i][0]

            alreadyStandingStill=True

        elif(speed < maxSpeedForStop and alreadyStandingStill):

            minutesStandingStill=minutesStandingStill+1

        elif(speed > maxSpeedForStop and alreadyStandingStill and minutesStandingStill > minTimeForStop):

            stops.append((pos[i][1],pos[i][2],firstStop,minutesStandingStill,pos[i][3]))

            alreadyStandingStill=False

            minutesStandingStill=0

            firstStop=""

        elif(speed > maxSpeedForStop and alreadyStandingStill and minutesStandingStill < minTimeForStop):

            alreadyStandingStill=False

            minutesStandingStill=0

            firstStop=""

    stops=pd.DataFrame(stops,columns=['lat','lon','stopStartedAt','stopDuration','vehicleId'])

    return(stops)



allStops=calcStops(inputData,minSpeed,minStopDuration)

print("%s number of stops found" % len(allStops))

print(allStops)

allStops.to_csv("allStops.csv")

print("Results written to allStops.csv")
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

print("Plotting distribution of all stops")

sns.distplot(a=allStops['stopDuration'], kde=False)
# Create heatmap of locations with stops and distribution of stop times

print("Plotting heatmap of identifed stops and displaying it centered around Gothenburg")





import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster





def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')



# Create a base map

m_5 = folium.Map(location=[57.7057749,11.972196], tiles='cartodbpositron', zoom_start=10)



# Add a heatmap to the base map

HeatMap(data=allStops[['lat', 'lon']], radius=10).add_to(m_5)



# Display the map

print("Map written to heatMapStops.html")

embed_map(m_5, 'heatMapStops.html')
minStopTime=20

stopsToCluster=allStops.loc[allStops.stopDuration > minStopTime]

print("Removed stops shorter than %s min (%s of %s remain to cluster)" % (minStopTime,len(stopsToCluster),len(allStops)))

print(stopsToCluster)

#Remove stops outside of Gothenburg area

print("Remove stops outside of Gothenburg area")

NWCorner=[57.965252, 11.388828]

SWCorner=[57.472575, 12.411249]

#print(stopsToCluster.loc[stopsToCluster.lat < 58])

stopsToCluster2=stopsToCluster.loc[stopsToCluster.lat < NWCorner[0]]

stopsToCluster2=stopsToCluster2.loc[stopsToCluster2.lon > NWCorner[1]]

stopsToCluster2=stopsToCluster2.loc[stopsToCluster2.lat > SWCorner[0]]

stopsToCluster2=stopsToCluster2.loc[stopsToCluster2.lon < SWCorner[1]]

stopsToCluster2=stopsToCluster2.reset_index()

stopsToCluster=stopsToCluster2.rename(columns={'index':'initialIndex'})

print(stopsToCluster)

print("Plotting distribution of stops with a duration of more than %s" % minStopTime)

sns.distplot(a=stopsToCluster['stopDuration'], kde=False)
# Create a base map

m_6 = folium.Map(location=[57.7057749,11.972196], tiles='cartodbpositron', zoom_start=10)



# Add a heatmap to the base map

HeatMap(data=stopsToCluster[['lat', 'lon']], radius=10).add_to(m_6)



# Display the map

print("Map written to heatMapStopsToCluster.html")

embed_map(m_6, 'heatMapStopsToCluster.html')

# Cluster stops to identify potential locations where vehicles are often standing still long enough to charge

# 

# Parameters:

# - Minimum standstill time: 20 min (shortest time where it would make sense to plug in)



minDistance=0.1 # Minimum distance between stops to be considered part of the same cluster



print("Clustering stops with a distance in between eachother of less than %s km" % minDistance)

# 

# Todo:

# - Add total number of vehicles stopping at identified locations

# - Add total number of stop time for vehicles stopping at identified locations



# Based on: https://github.com/gboeing/data-visualization/blob/master/location-history/google-location-history-cluster.ipynb

import os

import pandas as pd, numpy as np, matplotlib.pyplot as plt, time

from sklearn.cluster import DBSCAN

from geopy.distance import great_circle

from shapely.geometry import MultiPoint

from datetime import datetime as dt



def get_centermost_point(cluster):

    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)

    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)

    return tuple(centermost_point)



def dbscan_reduce(df, epsilon, x='lon', y='lat'):

    start_time = time.time()

    # represent points consistently as (lat, lon) and convert to radians to fit using haversine metric

    coords = df[[y,x]].values  

    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

    cluster_labels = db.labels_

    num_clusters = len(set(cluster_labels))

    print('Found {:,} clusters'.format(num_clusters))

    

    clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])

    clusterMembers=pd.DataFrame(db.labels_,columns=['clusterId'])



    # find the point in each cluster that is closest to its centroid

    centermost_points = clusters.map(get_centermost_point)



    # unzip the list of centermost points (lat, lon) tuples into separate lat and lon lists

    lats, lons = zip(*centermost_points)

    rep_points = pd.DataFrame({x:lons, y:lats})

    rep_points.tail()

    

    # pull row from original data set where lat/lon match the lat/lon of each row of representative points

    rs = rep_points.apply(lambda row: df[(df[y]==row[y]) & (df[x]==row[x])].iloc[0], axis=1)

    

    # all done, print outcome

    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'

#     print(message.format(len(df), len(rs), 100*(1 - float(len(rs)) / len(df)), time.time()-start_time))    

    return(rs,clusterMembers)



kms_per_radian = 6371.0088

eps_rad = minDistance / kms_per_radian

clusteredStops,clusterMembers = dbscan_reduce(stopsToCluster, epsilon=eps_rad)



clusteredStops=clusteredStops.join(clusterMembers.groupby('clusterId').clusterId.count())

clusteredStops=clusteredStops.rename(columns={'clusterId':'numberOfStops'})

clusteredStops=clusteredStops.drop(columns="stopStartedAt")

clusteredStops=clusteredStops.drop(columns="stopDuration")

clusteredStops=clusteredStops.drop(columns="vehicleId")



print(clusteredStops)

print(clusterMembers)

clusteredStops.to_csv("clusteredStops.csv")

#clusteredStops.to_csv("clusteredStops.csv")

print("Results written to clusteredStops.csv")

# Filter approrpiate locations for charging stations

# 

# Parameters:

# - Minimum number of stops: 20 (to obtain utilization degree (to be adjusted based on length of time analyzed))

minStops=20



print("Removing locations with less than %s stops" % minStops)



chargeStations=clusteredStops[clusteredStops.numberOfStops>minStops]



print(chargeStations)
# Rank charging stations with eigenvector centrality measure

import networkx as nx

print("Ranking charging stations with eigenvector centrality")

chargingStops=stopsToCluster.join(clusterMembers)

chargingStops=chargingStops.loc[chargingStops.clusterId > 0]



matrixSize=int(max(chargingStops.clusterId))

odMatrix=np.zeros((matrixSize+1,matrixSize+1))



prevStop=-1



for stop in chargingStops.iterrows(): 

    if(not prevStop==-1 and stop[1][5]==prevStop[1][5]):

        odMatrix[prevStop[1][6],stop[1][6]]=odMatrix[prevStop[1][6],stop[1][6]]+1

    prevStop=stop

G=nx.from_numpy_matrix(odMatrix)

centrality = nx.eigenvector_centrality(G)



for row in chargeStations.iterrows():

    chargeStations.loc[row[0],'score']=centrality[row[0]]



i=1

for row in chargeStations.sort_values(by='score',ascending=False).iterrows():

    chargeStations.loc[row[0],'rank']=i

    i=i+1

print(chargeStations.sort_values(by='score',ascending=False))

chargeStations.to_csv("chargeStations.csv")

print("Results written to chargeStations.csv")
print("Make map with markers for charging stations")



import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster







for idx, row in chargeStations.iterrows():



    Marker([row['lat'], row['lon']],tooltip='ID: %s<br>Rank: %s<br>Number of stops: %s<br>Score: %s' %(idx, row['rank'],row['numberOfStops'],row['score'])).add_to(m_6)

    

embed_map(m_6, 'heatMapStopsWithChargingStations.html')
# Filter and plot approrpiate locaations for charging stations

# 

# Parameters:

# - Minimum number of stops: 20 (to obtain utilization degree (to be adjusted based on length of time analyzed))



print("Adding markers for charging stations")



import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



m_2 = folium.Map(location=[57.7057749,11.972196], tiles='cartodbpositron', zoom_start=10)



for idx, row in chargeStations.iterrows():

    Marker([row['lat'], row['lon']],tooltip='ID: %s<br>Rank: %s<br>Number of stops: %s<br>Score: %s' %(row.name, row['rank'],row['numberOfStops'],row['score'])).add_to(m_2)

    

embed_map(m_2, 'mapWithChargingStations.html')