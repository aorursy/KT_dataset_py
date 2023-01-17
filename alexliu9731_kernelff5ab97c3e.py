#ECE-475-HW!
#Zheng Liu
#An analysis of Citibike service
#Based on the data of all the rides in July 2018
#Downloaded from Citibike official website
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import geopy as gp
from geopy import distance 
#!{sys.executable} -m pip install folium
import folium
#Read Data
citi = pd.read_csv('/Users/Alex/Documents/ML/201807-citibike-tripdata.csv')
print("Total number of use is {}.".format(citi.shape[0]))
citi.head()
#Age Distribution
sns.set(color_codes=True)
sns.distplot(2018-citi['birth year'], kde = False, rug = False);
plt.xticks(np.arange(0, 140, 10))
plt.ylabel("Count")
plt.xlabel("Age")
plt.title('Distribution by age')
plt.show()
#Gender
#sns.distplot(citi['gender'], kde = False, rug = False);
sns.countplot(x="gender", data=citi)
plt.xticks(np.arange(3), ('Unknown', 'Male', 'Female'))
plt.ylabel("Count")
plt.title('Distribution by Gender')
plt.show()
sns.countplot(x="usertype", data=citi)
plt.ylabel("Count")
plt.title('Distribution by User Type')
plt.show()
#Gener vs duration
sns.kdeplot(citi[(citi['gender']==1) & (citi['tripduration']<7200)]['tripduration'], shade=True)
sns.kdeplot(citi[(citi['gender']==2) & (citi['tripduration']<7200)]['tripduration'], shade=True)
plt.title('Distribution of Number of Uses of Stations')
plt.xlabel('duration (seconds)')
plt.yticks([])
plt.legend(['Male', 'Female'])
#sns.relplot(x="gender", y="tripduration", data = citi[citi.tripduration<7200]);
plt.show()
#geo stuff

#If start_station == end_station, exclude
citi['trip'] = citi[citi['start station id']!=citi['end station id']][['start station id','end station id']].apply(tuple,axis=1)

count = citi['trip'].value_counts()
citi['trip'].drop_duplicates();
print("Below is the list of trips (combination of stations basically)\n\n")
print(count)
print('\n\n\n\n\n\n')

popular = count.idxmax()
stationsjson = pd.read_json('/Users/Alex/Documents/ML/stations.json');
stations = pd.io.json.json_normalize(stationsjson['stationBeanList']);
sstation=stations[stations.id == popular[0]]['stAddress1']
estation=stations[stations.id == popular[1]]['stAddress1']


print("\nStart station of the most popular trip is " + str(sstation).split('\n')[0])
print("\nEnd station of the most popular trip is " + str(estation).split('\n')[0])
loc1=stations[stations.id == popular[0]][['latitude','longitude']].apply(tuple,axis=1)
loc2=stations[stations.id == popular[1]][['latitude','longitude']].apply(tuple,axis=1)
dis = distance.distance(loc1,loc2).miles
print("\n\n\nThe distance between them is {:.2f} mile".format(dis))


print("\n\n\nI don't know where they are. So I included a map to show them.")
loc_1 = loc1.reset_index().rename(columns={0:'A'})
loc_2 = loc1.reset_index().rename(columns={0:'A'})

m = folium.Map(location=[40.75, -74], tiles="Mapbox Bright", zoom_start=11)
folium.Marker(loc_1.at[0,'A'], "Start").add_to(m)
folium.Marker(loc_2.at[0,'A'], "End").add_to(m)
m 
#geo continued
count2 = citi['start station id'].value_counts()
count2.reset_index().plot(y = 1, legend=None)
plt.title('Distribution of Number of Uses of Stations')
plt.ylabel('# of use')
plt.xticks([])
plt.show()

print("\nBelow is the list of stations with their id on the left and # of use on the right\n")
count2

id1 = count2.idxmax()
id2 = count2[count2>1].idxmin()

most_popular = stations[stations.id == id1]['stAddress1']
least_popular = citi[citi['start station id']==id2].reset_index().loc[0].at['start station name']

print("\nMost popular station is " + str(most_popular).split('\n')[0])
print("\nLeast popular station is " + str(least_popular).split('\n')[0])

m = folium.Map(location=[40.75, -74], tiles="Mapbox Bright", zoom_start=11)
folium.Marker((citi[citi['start station id']==id1].reset_index().loc[0].at['start station latitude'], citi[citi['start station id']==id1].reset_index().loc[0].at['start station longitude']), "Most Popular").add_to(m)
folium.Marker((citi[citi['start station id']==id2].reset_index().loc[0].at['start station latitude'], citi[citi['start station id']==id2].reset_index().loc[0].at['start station longitude']), "Least Popular").add_to(m)
m 


#Efficiency of docks
print("\nCompare the most popular stations and the ones with the most docks, emmmmm\n")
dockN = stations.sort_values(by='totalDocks',ascending = False)['totalDocks'].reset_index()[0:10]
compare = count2.reset_index()[0:10].join(dockN, lsuffix='usage#', rsuffix='dock#')
compare.rename(columns={"indexusage#": "station id", "start station id": "usage#", "indexdock#":"station id", "totalDocks":"dock#"})

#Some jokes
print("The oldest rider is",2018-citi['birth year'].min(),"born in", citi['birth year'].min())

print("The longest ride is", citi['tripduration'].max(), "seconds,", citi['tripduration'].max()/3600, "hours,", citi['tripduration'].max()/3600/24, "days (in a month )" )

