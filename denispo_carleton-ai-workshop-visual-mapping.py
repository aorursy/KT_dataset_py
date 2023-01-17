#import the various libraries we require for the analysis 

import numpy as np

import pandas as pd

from numpy import array

import matplotlib as mpl



#import the tools we need to plot maps

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm #cm is a tool that allows us to gain access to a predefined set of colours in matplotlib
cities = pd.read_csv("../input/top-500-indian-cities/cities_r2.csv")

#print the data for the first five cities in the data set

cities.head()
#prints an overview of each columns with the number of entries in each column and the data type

cities.info()
#gives an even more detailed view into each column

cities.describe()
# Create a bar graph of the the number of cities taken for analysis per state 



fig = plt.figure(figsize=(20,20))



#count the number of states, sort in ascending order

states = cities.groupby('state_name')['name_of_city'].count().sort_values(ascending=True)



states.plot(kind='barh', fontsize=18)

plt.grid(b=True, which='both', color='black', linestyle = '-')

plt.xlabel('No. of cities taken for analysis', fontsize = 20)

















#We split the data in location using the delimiter ',' and since the first index i.e., [0] is the latitude coordinate,

#we store and we did the same for longitude

cities['latitude']=cities['location'].apply(lambda x:x.split(',')[0])

cities['longitude']=cities['location'].apply(lambda x:x.split(',')[1])
#sort the cities according to population

top_pop_cities=cities.sort_values(by='population_total',ascending=False)



#store the top 10 cities under a separate name

top10_pop_cities=top_pop_cities.head(10)

top10_pop_cities
#setting up Basemap for our first map! 



fig = plt.figure(figsize=(20,15))



themap = Basemap(width=1200000, height=900000, projection = 'lcc', resolution='l', llcrnrlon=67, llcrnrlat=5, urcrnrlon=99, urcrnrlat=37, lat_0=22, lon_0=78)



themap.bluemarble()

themap.drawmapboundary()

themap.drawcountries()

themap.drawcoastlines()



lg = array(top10_pop_cities['longitude'])

lt = array(top10_pop_cities['latitude'])



pt = array(top10_pop_cities['population_total'])

nc = array(top10_pop_cities['name_of_city'])



x,y = themap(lg, lt)



population_size = top10_pop_cities['population_total'].apply(lambda x:int(x/5000))



themap.scatter(x, y, s=population_size, c=population_size, marker='o', cmap=cm.Dark2, alpha=0.7)



for ncs,xpt,ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize = 12, fontweight = 'bold')

    

plt.title('Top 10 most populous cities in India', fontsize = 18)



#since it is tedious to write the code for drawing a map every single time, let us go ahead and write a function that draws a map

#we can then call this function every single time we require a map



def plot_map(sizes, colorbarValue):

    plt.figure(figsize=(19, 20))

    f, ax = plt.subplots(figsize=(19,20))

    

    themap = Basemap(width=1200000, height=900000, projection = 'lcc', resolution='l', llcrnrlon=67, llcrnrlat=5, urcrnrlon=99, urcrnrlat=37, lat_0=22, lon_0=78, ax=ax)

    

    themap.drawmapboundary()

    themap.drawcountries()

    themap.drawcoastlines()

    x, y = themap(array(cities['longitude']), array(cities['latitude']))

    

    cs = themap.scatter(x,y, s=sizes, c=sizes, marker='o', cmap=cm.get_cmap('Dark2'), alpha=0.8)



    cbar = themap.colorbar(cs, location='right', pad ="2%")

    cbar.ax.set_yticklabels(colorbarValue)
population_sizes=cities["population_total"].apply(lambda x:int(x/5000))



#setting the lowest and highest value on the colorbar

colorbarValue=np.linspace(cities['population_total'].min(),cities['population_total'].max())



#converting the value on the legend from float to int

colorbarValue=colorbarValue.astype(int)



plot_map(population_sizes,colorbarValue)
#specifying the size of the figure

fig=plt.figure(figsize=(20,20))



#grouping states according to the total number of total graduates in that state

states=cities.groupby('state_name')['total_graduates'].sum().sort_values(ascending=True)



#specifying the type of graph

states.plot(kind="barh", fontsize = 20)



#b = to show the grid lines

plt.grid(b=True, which='both', color='Black',linestyle='-')



#label the x-axis

plt.xlabel('Total graduates per state', fontsize = 20)
#ranking the cities according to the total graduates in the city

top_total_graduates_cities = cities.sort_values(by='total_graduates',ascending=False)



#separating the top 10 cities with the highest number of total graduates

top10_total_graduates_cities=top_total_graduates_cities.head(10)

top10_total_graduates_cities
plt.subplots(figsize=(20, 15))

themap = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



#map.bluemarble()



themap.drawmapboundary ()

themap.drawcountries()

themap.drawcoastlines ()







lg=array(top10_total_graduates_cities['longitude'])

lt=array(top10_total_graduates_cities['latitude'])

pt=array(top10_total_graduates_cities['total_graduates'])

nc=array(top10_total_graduates_cities['name_of_city'])





# START coding here 

x, y = themap(lg,lt)



population_size_total_graduates = top10_total_graduates_cities['total_graduates'].apply(lambda x:int(x/5000))





plt.scatter(x, y, s=population_size_total_graduates, marker='o', c=population_size_total_graduates, cmap = cm.get_cmap('Dark2'), alpha=0.7)



for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

    

plt.title('Top 10 cities in India with total graduates', fontsize=20)







population_sizes = cities["total_graduates"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["total_graduates"].min(), cities["total_graduates"].max())

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)