# importing packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from numpy import array

import matplotlib as mpl



# for plots

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.dates import date2num

from mpl_toolkits.basemap import Basemap



# for date and time processing

import datetime



# for statistical graphs

import seaborn as sns
cities = pd.read_csv ("../input/cities_r2.csv")
cities.head ()
cities.info ()

# there is no null values anywhere in the dataset
cities.describe ()

print (cities.describe(include=['O']))

# from the below output, we can learn that there is two Aurangabad's. One is in Maharashtra 

# and one is in Bihar

# most of the cities are selected from Uttar Pradesh
# A bar chart to show from which states, how many cities are taken for examination.

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['name_of_city'].count().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of cities taken for analysis', fontsize = 20)

plt.show ()

# we can see states like UP and WB are given high priority by taking more than 60 cities.
# Extracting Co-ordinates details from the provided data

cities['latitude'] = cities['location'].apply(lambda x: x.split(',')[0])

cities['longitude'] = cities['location'].apply(lambda x: x.split(',')[1])

cities.head(1)
# A table to show top 10 cities with most population

print("The Top 10 Cities sorted according to the Total Population (Descending Order)")

top_pop_cities = cities.sort_values(by='population_total',ascending=False)

top10_pop_cities=top_pop_cities.head(10)

top10_pop_cities
# Plotting these top 10 populous cities on India map. Circles are sized according to the 

# population of the respective city



plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_pop_cities['longitude'])

lt=array(top10_pop_cities['latitude'])

pt=array(top10_pop_cities['population_total'])

nc=array(top10_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes = top10_pop_cities["population_total"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes, marker="o", c=population_sizes, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Populated Cities in India',fontsize=20)
# A bar chart to show the population of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['population_total'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of cities taken for analysis', fontsize = 20)

plt.show ()

# we can see states like Maharashtra and UP have huge urban population
# Creating a function to plot the population data on real India map



def plot_map(sizes, colorbarValue):



    plt.figure(figsize=(19,20))

    f, ax = plt.subplots(figsize=(19, 20))



    # Setting up Basemap

    map = Basemap(width=5000000, height=3500000, resolution='l', projection='aea', llcrnrlon=69,

                  llcrnrlat=6, urcrnrlon=99, urcrnrlat=36, lon_0=78, lat_0=20, ax=ax)

                  

    # draw map boundaries

    map.drawmapboundary()

    map.drawcountries()

    map.drawcoastlines()



    # plotting cities on map using previously derived coordinates

    x, y = map(array(cities["longitude"]), array(cities["latitude"]))

    cs = map.scatter(x, y, s=sizes, marker="o", c=sizes, cmap=cm.Dark2, alpha=0.5)



    # adding colorbar

    cbar = map.colorbar(cs, location='right',pad="5%")

    cbar.ax.set_yticklabels(colorbarValue)



    plt.show()
# Using the function created in the previous cell, we are plotting the population data



population_sizes = cities["population_total"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["population_total"].min(), cities["population_total"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)
# A bar chart to show the male population of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['population_male'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of cities taken for analysis', fontsize = 20)

plt.show ()

# we can see states like Maharashtra and UP have huge male population
# Plotting the same on the map

population_sizes = cities["population_male"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["population_male"].min(), cities["population_male"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)
# A table to show top 10 cities with most male population

print("The Top 10 Cities sorted according to the Total Male Population (Descending Order)")

top_male_cities = cities.sort_values(by='population_male',ascending=False)

top10_male_pop_cities=top_male_cities.head(10)

top10_male_pop_cities
# Plotting these top 10 male populous cities on India map. Circles are sized according to the 

# male population of the respective city



plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_male_pop_cities['longitude'])

lt=array(top10_male_pop_cities['latitude'])

pt=array(top10_male_pop_cities['population_male'])

nc=array(top10_male_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes_male = top10_male_pop_cities["population_male"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes_male, marker="o", c=population_sizes_male, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Male Populated Cities in India',fontsize=20)
# A bar chart to show the female population of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['population_female'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of cities taken for analysis', fontsize = 20)

plt.show ()

# we can see again states like Maharashtra and UP have huge female population
# Plotting the same on the map

population_sizes = cities["population_female"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["population_female"].min(), cities["population_female"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)
# A table to show top 10 cities with most female population

print("The Top 10 Cities sorted according to the Total Female Population (Descending Order)")

top_female_cities = cities.sort_values(by='population_female',ascending=False)

top10_female_pop_cities=top_female_cities.head(10)

top10_female_pop_cities
# Plotting these top 10 female populous cities on India map. Circles are sized according to the 

# female population of the respective city



plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_female_pop_cities['longitude'])

lt=array(top10_female_pop_cities['latitude'])

pt=array(top10_female_pop_cities['population_female'])

nc=array(top10_female_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes_female = top10_female_pop_cities["population_female"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes_female, marker="o", c=population_sizes_female, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Female Populated Cities in India',fontsize=20)
# A bar chart to show the kids population of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['0-6_population_total'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of kids', fontsize = 20)

plt.show ()

# we can see again states like Maharashtra and UP have huge kids population living in cities
# Plotting the same on the map

population_sizes = cities["0-6_population_total"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["0-6_population_total"].min(), cities["0-6_population_total"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)

# Kids population is obviously smaller than the overall population and bigger cities like Delhi,

# Mumbai, Banglore, Kolkata, Hyderabad, Chennai have vast number of kids living in cities
# Lets find the top ten cities in which large number of kids live

print("The Top 10 Cities sorted according to the Total Kids Population (Descending Order)")

top_kids_cities = cities.sort_values(by='0-6_population_total',ascending=False)

top10_kids_pop_cities=top_kids_cities.head(10)

top10_kids_pop_cities
# Lets find the top ten cities in which large number of kids live



plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_kids_pop_cities['longitude'])

lt=array(top10_kids_pop_cities['latitude'])

pt=array(top10_kids_pop_cities['0-6_population_total'])

nc=array(top10_kids_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes_kids = top10_kids_pop_cities["0-6_population_total"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes_kids, marker="o", c=population_sizes_kids, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Kids Populated Cities in India',fontsize=20)
# A bar chart to show the male kids population of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['0-6_population_male'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of male kids', fontsize = 20)

plt.show ()

# we can see again states like Maharashtra and UP have huge male kids population living in cities
# Plotting the same on the map

population_sizes = cities["0-6_population_male"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["0-6_population_male"].min(), cities["0-6_population_male"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)

# Kids population is obviously smaller than the overall population and bigger cities like Delhi,

# Mumbai, Banglore, Kolkata, Hyderabad, Chennai have vast number of kids living in cities
# Lets find the top ten cities in which large number of male kids live

print("The Top 10 Cities sorted according to the Total Male Kids Population (Descending Order)")

top10_male_kids_cities = cities.sort_values(by='0-6_population_male',ascending=False)

top10_male_kids_pop_cities=top10_male_kids_cities.head(10)

top10_male_kids_pop_cities
# Lets find the top ten cities in which large number of male kids live



plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_male_kids_pop_cities['longitude'])

lt=array(top10_male_kids_pop_cities['latitude'])

pt=array(top10_male_kids_pop_cities['0-6_population_male'])

nc=array(top10_male_kids_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes_male_kids = top10_male_kids_pop_cities["0-6_population_male"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes_male_kids, marker="o", c=population_sizes_male_kids, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Male Kids Populated Cities in India',fontsize=20)
# A bar chart to show the female kids population of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['0-6_population_female'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of female kids', fontsize = 20)

plt.show ()

# we can see again states like Maharashtra and UP have huge male kids population living in cities
# Plotting the same on the map

population_sizes = cities["0-6_population_female"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["0-6_population_female"].min(), cities["0-6_population_female"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)

# Kids population is obviously smaller than the overall population and bigger cities like Delhi,

# Mumbai, Banglore, Kolkata, Hyderabad, Chennai have vast number of kids living in cities
# Lets find the top ten cities in which large number of female kids live

print("The Top 10 Cities sorted according to the Total Female Kids Population (Descending Order)")

top10_female_kids_cities = cities.sort_values(by='0-6_population_female',ascending=False)

top10_female_kids_pop_cities=top10_female_kids_cities.head(10)

top10_female_kids_pop_cities
# Lets find the top ten cities in which large number of female kids live



plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_female_kids_pop_cities['longitude'])

lt=array(top10_female_kids_pop_cities['latitude'])

pt=array(top10_female_kids_pop_cities['0-6_population_female'])

nc=array(top10_female_kids_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes_female_kids = top10_female_kids_pop_cities["0-6_population_female"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes_female_kids, marker="o", c=population_sizes_female_kids, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Female Kids Populated Cities in India',fontsize=20)
# A bar chart to show the total literates of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['literates_total'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('Total litracy rate of states', fontsize = 20)

plt.show ()

# we can see again states like Maharashtra and UP have huge litrate population living in cities

# Plotting the same on the map

population_sizes = cities["literates_total"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["literates_total"].min(), cities["literates_total"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)

# Major metro cities again shows higher litracy rates
# Lets find the top ten cities in which large number of literates live

print("The Top 10 Cities sorted according to the Total litrate Population (Descending Order)")

top10_literate_cities = cities.sort_values(by='literates_total',ascending=False)

top10_literate_cities=top10_literate_cities.head(10)

top10_literate_cities
# lets plot the top 10 literate cities on India map

plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_female_kids_pop_cities['longitude'])

lt=array(top10_female_kids_pop_cities['latitude'])

pt=array(top10_female_kids_pop_cities['literates_total'])

nc=array(top10_female_kids_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes_female_kids = top10_female_kids_pop_cities["literates_total"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes_female_kids, marker="o", c=population_sizes_female_kids, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 most literate Cities in India',fontsize=20)
# # A bar chart to show the total male literates of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['literates_male'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of total male literates of the states', fontsize = 20)

plt.show ()

# we can see again states like Maharashtra and UP have huge male literate population living in cities
# Plotting the same on the map

population_sizes = cities["literates_male"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["literates_male"].min(), cities["literates_male"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)

# Major metro cities again shows higher male litracy rates
# Lets find the top ten cities in which large number of males are literate

print("The Top 10 Cities sorted according to the male literate Population (Descending Order)")

top10_male_literate_cities = cities.sort_values(by='literates_male',ascending=False)

top10_male_literate_cities=top10_male_literate_cities.head(10)

top10_male_literate_cities
# Lets find the top ten cities in which large number of males are literate on the map of India



plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_female_kids_pop_cities['longitude'])

lt=array(top10_female_kids_pop_cities['latitude'])

pt=array(top10_female_kids_pop_cities['literates_male'])

nc=array(top10_female_kids_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes_female_kids = top10_female_kids_pop_cities["literates_male"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes_female_kids, marker="o", c=population_sizes_female_kids, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 male litracy cities in India',fontsize=20)
# A bar chart to show the female litracy population of the states

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['literates_female'].sum().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of female literates', fontsize = 20)

plt.show ()

# we can see again states like Maharashtra and UP have huge female literate population living in cities
# Plotting the same on the map

population_sizes = cities["literates_female"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(cities["literates_female"].min(), cities["literates_female"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)



plot_map(population_sizes, colorbarValue)

# Major metro cities again shows higher female litracy rates
# Lets find the top ten cities in which large number of female literates live

print("The Top 10 Cities sorted according to the Total Female literates Population (Descending Order)")

top10_female_literates_cities = cities.sort_values(by='literates_female',ascending=False)

top10_female_literates_cities = top10_female_literates_cities.head(10)

top10_female_literates_cities
# Lets find the top ten cities in which large number of female literates live



plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_female_kids_pop_cities['longitude'])

lt=array(top10_female_kids_pop_cities['latitude'])

pt=array(top10_female_kids_pop_cities['literates_female'])

nc=array(top10_female_kids_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes_female_kids = top10_female_kids_pop_cities["literates_female"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes_female_kids, marker="o", c=population_sizes_female_kids, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Female literates Populated Cities in India',fontsize=20)
# seperating effective literacy rate from the main dataset and sorting then in descending order

state_literacy_effective = cities[["state_name","effective_literacy_rate_total","effective_literacy_rate_male","effective_literacy_rate_female"]].groupby("state_name").agg({"effective_literacy_rate_total":np.average,

                                                                                                "effective_literacy_rate_male":np.average,

                                                                                                "effective_literacy_rate_female":np.average})

state_literacy_effective.sort_values("effective_literacy_rate_total", ascending=True).plot(kind="barh",

                      grid=True,

                      figsize=(16,15),

                      alpha = 0.6,

                      width=0.6,

                      stacked = False,

                      edgecolor="g",

                      fontsize = 20)

plt.grid(b=True, which='both', color='lightGreen',linestyle='-')

plt.show ()

# from the below chart, Mizoram, Kerala and HP have highest effective literacy rate across India
# seperating Graduates from the main dataset and sorting then in descending order

state_graduates  = cities[["state_name",

                                  "total_graduates",

                                  "male_graduates",

                                  "female_graduates"]].groupby("state_name").agg({"total_graduates":np.average,

                                                                                  "male_graduates":np.average,

                                                                                  "female_graduates":np.average})

# Plotting the bar chart 

state_graduates.sort_values("total_graduates", ascending=True).plot(kind="barh",

                      grid=True,

                      figsize=(16,15),

                      alpha = 0.6,

                      width=0.6,

                      stacked = False,

                      edgecolor="g",

                      fontsize = 20)

plt.grid(b=True, which='both', color='lightGreen',linestyle='-')

plt.show ()

# from the below Chandigarh, NCT of Delhi, Maharashta have most of their graduates living in cities.

# we can note that Kerala and Meghalaya are the only states that have more number of female graduates than 

# male graduates
# A bar chart to show how many females are there for per 1000 males.

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['sex_ratio'].mean().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of females available for every 1000 males', fontsize = 20)

plt.show ()

# We can see that states of Kerala, Manipur, Meghalaya, Puducherry, Mizoram are having more females per 1000 males
# A bar chart to show how many females are there for per 1000 males.

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['child_sex_ratio'].mean().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of girls available for every 1000 boys', fontsize = 20)

plt.show ()

# Not even a single state have 1000 girls for every 1000 boys