import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from numpy import array

from mpl_toolkits.basemap import Basemap

from functools import reduce

from subprocess import check_output



print(check_output(["ls", "../input/cities_r2.csv"]).decode("utf8"))

inputfile = pd.read_csv('../input/cities_r2.csv')

inputfile.head(2)
print("The number of states in India are: ",inputfile['state_code'].nunique()) 

inputfile['latitude'] = inputfile['location'].apply(lambda x: x.split(',')[0])

inputfile['longitude'] = inputfile['location'].apply(lambda x: x.split(',')[1])

inputfile.head(1)
print("The Top 10 Cities sorted according to the Total Population (Descending Order)")

top_pop_cities = inputfile.sort_values(by='population_total',ascending=False)

top10_pop_cities=top_pop_cities.head(10)

top10_pop_cities


plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary(fill_color='lightblue')

map.fillcontinents(color='orange')

map.drawcountries(color='black')

map.drawcoastlines(linewidth=0.5,color='black')  



lg=array(top10_pop_cities['longitude'])

lt=array(top10_pop_cities['latitude'])

pt=array(top10_pop_cities['population_total'])

nc=array(top10_pop_cities['name_of_city'])



x, y = map(lg, lt)

plt.plot(x, y, 'ro', markersize=10)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Populated Cities in India',fontsize=20)
print("The Top 10 Cities sorted according to the Literacy Rate (Descending Order)")

top10_lit_cities = inputfile.sort_values(by='effective_literacy_rate_total',ascending=False).head(10)

top10_lit_cities
plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary(fill_color='lightblue')

map.fillcontinents(color='green')

map.drawcountries(color='black')

map.drawcoastlines(linewidth=0.5,color='black')  



lg=array(top10_lit_cities['longitude'])

lt=array(top10_lit_cities['latitude'])

pt=array(top10_lit_cities['population_total'])

nc=array(top10_lit_cities['name_of_city'])



x, y = map(lg, lt)

plt.plot(x, y, 'ro', markersize=10)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Literate Cities in India',fontsize=20)
print("The States with the number of Top Cities in them")

states_no_of_top_cities=inputfile["state_name"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(25, 10))

states_no_of_top_cities.plot(title='States with the number of Top Cities in them',kind="bar", fontsize=20)
print("The States with the Average Literacy Rate of their Top Cities")

litratesort=inputfile.groupby(['state_name'])['effective_literacy_rate_total'].mean().sort_values(ascending=False)

plt.figure(figsize=(25, 10))

litratesort.head(29).plot(title='Average Literacy Rate of States',kind='bar', fontsize=20)
print("The Population of States considering the Top Cities in it")

pop_state=inputfile.groupby(['state_name'])['population_total'].sum().sort_values(ascending=False)

plt.figure(figsize=(25, 10))

pop_state.head(29).plot(title='Population of States considering the Top Cities in it',kind='bar', fontsize=20)
print("The Average Sex Ratio of the States considering all the Top Cities in it")

sex_ratio_states=inputfile.groupby(['state_name'])['sex_ratio'].mean().sort_values(ascending=False)

plt.figure(figsize=(25, 10))

sex_ratio_states.head(29).plot(title='Average Sex Ratio of the States considering all the Top Cities in it',kind='bar', fontsize=20)
inputfile['graduate_ratio']=inputfile['total_graduates']/(inputfile['population_total']-inputfile['0-6_population_total'])

inputfile.head(2)
print("The Graduates Ratio of the States considering all the Top Cities in it")

grad_ratio_states=inputfile.groupby(['state_name'])['graduate_ratio'].mean().sort_values(ascending=False)

plt.figure(figsize=(25, 10))

grad_ratio_states.head(29).plot(title='Graduates Ratio of the States considering all the Top Cities in it',kind='bar', fontsize=20)
print("Top States having better Total Literacy Rates, Sex-Ratio and Graduation Ratio are as follows:")

los = [litratesort.head(15),sex_ratio_states.head(15),grad_ratio_states.head(15)]

pd.concat(los, axis=1, join='inner')