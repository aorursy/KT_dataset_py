import pandas as pd

import numpy as np
cities=pd.read_csv("../input/top-500-indian-cities/cities_r2.csv")
cities.head()
cities.describe()
cities.info()
from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

corr=cities.corr()

f, ax = plt.subplots(figsize=(6,6))

sns.heatmap(corr, vmax=.8, square=True);
cities.describe(include='O')
fig = plt.figure(figsize=(20,10))

states = cities.groupby('state_name')['name_of_city'].count().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20,color='green')

plt.grid(b=True, which='both', color='black')

plt.xlabel('No of cities taken from each state', fontsize = 20)

plt.show ()
lit_by_states  = cities.groupby('state_name').agg({'literates_total': np.sum})

pop_by_states  = cities.groupby('state_name').agg({'population_total': np.sum})

literate_rate = lit_by_states.literates_total * 100 / pop_by_states.population_total

literate_rate = literate_rate.sort_values(ascending=True)



plt.subplots(figsize=(7, 6))

ax = sns.barplot(x=literate_rate, y=literate_rate.index,color='brown')

ax.set_title('States according to literacy rate', size=20, alpha=0.5, color='red')

ax.set_xlabel('Literacy Rate(as % of population)', size=15, alpha=0.5, color='red')

ax.set_ylabel('States', size=15, alpha=0.5, color='red')
cities.kurt()
cities.skew()
cities['lattitude'] = cities['location'].apply(lambda x: x.split(',')[0])

cities['longitude'] = cities['location'].apply(lambda x: x.split(',')[1])

cities.head(5)
top_pop_cities = cities.sort_values(by='population_total',ascending=False)

top20_populated_cities=top_pop_cities.head(20)

top20_populated_cities
from matplotlib import cm

from matplotlib.dates import date2num

from mpl_toolkits.basemap import Basemap
plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



longitude=np.array(top20_populated_cities['longitude'])

lattitude=np.array(top20_populated_cities['lattitude'])

total_population=np.array(top20_populated_cities['population_total'])

city_name=np.array(top20_populated_cities['name_of_city'])



x, y = map(longitude, lattitude)

population_sizes = top20_populated_cities["population_total"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes, marker="o", c=population_sizes, cmap=cm.Dark2, alpha=0.8)





for ncs, xpt, ypt in zip(city_name, x, y):

    plt.text(xpt+6000, ypt+3000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 20 Populated Cities in India',fontsize=20)
def plot_map(sizes, colorbarValue):



    plt.figure(figsize=(19,20))

    f, ax = plt.subplots(figsize=(19, 20))

    map = Basemap(width=5000000, height=3500000, resolution='l', projection='aea', llcrnrlon=69,llcrnrlat=6, urcrnrlon=99, urcrnrlat=36, lon_0=78, lat_0=20, ax=ax)

    map.drawmapboundary()

    map.drawcountries()

    map.drawcoastlines()

    x, y = map(np.array(cities["longitude"]), np.array(cities["lattitude"]))

    cs = map.scatter(x, y, s=sizes, marker="o", c=sizes, cmap=cm.Dark2, alpha=0.5)

    cbar = map.colorbar(cs, location='right',pad="5%")

    cbar.ax.set_yticklabels(colorbarValue)

    plt.show()


population_sizes = cities["population_total"].apply(lambda x: int(x /6000))

colorbarValue = np.linspace(cities["population_total"].min(), cities["population_total"].max(),num=10)

colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)

population_sizes = cities["population_male"].apply(lambda x: int(x /6000))

colorbarValue = np.linspace(cities["population_male"].min(), cities["population_male"].max(),num=10)

colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
population_sizes = cities["population_female"].apply(lambda x: int(x / 6000))

colorbarValue = np.linspace(cities["population_female"].min(), cities["population_female"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)
population_sizes = cities["0-6_population_total"].apply(lambda x: int(x /6000))

colorbarValue = np.linspace(cities["0-6_population_total"].min(), cities["0-6_population_total"].max(),num=10)

colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
population_sizes = cities["0-6_population_male"].apply(lambda x: int(x /6000))

colorbarValue = np.linspace(cities["0-6_population_male"].min(), cities["0-6_population_male"].max(),num=10)

colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
population_sizes = cities["0-6_population_female"].apply(lambda x: int(x /6000))

colorbarValue = np.linspace(cities["0-6_population_female"].min(), cities["0-6_population_female"].max(),num=10)

colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
population_sizes = cities["literates_total"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["literates_total"].min(), cities["literates_total"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)
population_sizes = cities["literates_male"].apply(lambda x: int(x /6000))

colorbarValue = np.linspace(cities["literates_male"].min(), cities["literates_male"].max(),num=10)

colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
population_sizes = cities["literates_female"].apply(lambda x: int(x /6000))

colorbarValue = np.linspace(cities["literates_female"].min(), cities["literates_female"].max(),num=10)

colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
state_literacy_effective = cities[["state_name","effective_literacy_rate_total","effective_literacy_rate_male","effective_literacy_rate_female"]].groupby("state_name").agg({"effective_literacy_rate_total":np.average,"effective_literacy_rate_male":np.average,"effective_literacy_rate_female":np.average})

state_literacy_effective.head()
state_literacy_effective.shape
state_literacy_effective_sort= state_literacy_effective.sort_values("effective_literacy_rate_total", ascending=True)

state_literacy_effective_sort.plot(kind="barh",grid=True,figsize=(16,15),alpha = 0.6,width=0.6,stacked = False,edgecolor="g",fontsize = 20)

plt.grid(b=True, which='both')

plt.legend()

plt.show ()
state_graduates  = cities[["state_name","total_graduates","male_graduates","female_graduates"]].groupby("state_name").agg({"total_graduates":np.average,"male_graduates":np.average,"female_graduates":np.average})

state_graduates.sort_values("total_graduates", ascending=True).plot(kind="barh",grid=True,figsize=(16,15),alpha = 0.6,width=0.6,stacked = False,edgecolor="g",fontsize = 20)

plt.grid(b=True, which='both')

plt.legend()

plt.show ()
import geopandas as gpd

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster
uttarpradesh = cities[((cities.state_name == 'UTTAR PRADESH'))]

uttarpradesh.shape
m_1 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in uttarpradesh.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_1)



# Display the map

m_1
graduates_above_1lakh = cities[((cities.total_graduates > 100000))]

graduates_above_1lakh.shape
m_2 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in graduates_above_1lakh.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_2)



# Display the map

m_2
effective_literacy_rate_above_90 = cities[((cities.effective_literacy_rate_total > 90))]

effective_literacy_rate_above_90
m_3 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in effective_literacy_rate_above_90.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_3)



# Display the map

m_3
female_literates_above = cities[((cities.literates_female >160000))]

m_4 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in female_literates_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_4)



# Display the map

m_4
male_literates_above = cities[((cities.literates_male >160000))]

m_5 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in male_literates_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_5)



# Display the map

m_5
literates_mean=cities.literates_total.mean()
literates_above = cities[((cities.literates_male >literates_mean))]

m_6 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in literates_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_6)



# Display the map

m_6
total_kids_above = cities[((cities['0-6_population_total'] >50000))]

m_7 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in total_kids_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_7)



# Display the map

m_7
male_kids_above = cities[((cities['0-6_population_male'] >50000))]

m_8 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in male_kids_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_8)



# Display the map

m_8
female_kids_above = cities[((cities['0-6_population_female'] >50000))]

m_9 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in female_kids_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_9)



# Display the map

m_9
total_pop_above = cities[((cities.population_total >2000000))]

m_10 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in total_pop_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_10)



# Display the map

m_10
male_pop_above = cities[((cities.population_male >2000000))]

m_11 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in male_pop_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_11)



# Display the map

m_11
female_pop_above = cities[((cities.population_female >2000000))]

m_12 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in female_pop_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_12)



# Display the map

m_12
female_grad_above = cities[((cities.female_graduates >200000))]

m_13 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in female_grad_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_13)



# Display the map

m_13
male_grad_above = cities[((cities.male_graduates >200000))]

m_14 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in male_grad_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_14)



# Display the map

m_14
child_sex_ratio_above = cities[((cities.child_sex_ratio >950))]

m_15 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in child_sex_ratio_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_15)



# Display the map

m_15
sex_ratio_above = cities[((cities.sex_ratio >980))]

m_16 = folium.Map(location=[20.5936832,78.962883], tiles='cartodbpositron', zoom_start=5)



# Add points to the map

for idx, row in sex_ratio_above.iterrows():

    Marker([row['lattitude'], row['longitude']]).add_to(m_16)



# Display the map

m_16