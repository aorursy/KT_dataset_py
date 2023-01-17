import pandas as pd

import numpy as np

from numpy import array

import seaborn as sns

import matplotlib as mpl

from matplotlib import cm

import matplotlib.pyplot as plt

import datetime

%matplotlib inline
df = pd.read_csv ("../input/cities_r2.csv")
df.head()
df.info()
df.describe()
df.shape
# Extracting Co-ordinates details from the provided data

df['latitude'] = df['location'].apply(lambda x: x.split(',')[0])

df['longitude'] = df['location'].apply(lambda x: x.split(',')[1])

df.head()
pop_cities = df.sort_values(by = 'population_total', ascending = False)

ten_pop_cities = pop_cities.head(10)

ten_pop_cities
from mpl_toolkits.basemap import Basemap



plt.subplots(figsize=(20, 15))



map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

#set a background colour

map.drawmapboundary(fill_color='#85A6D9')



# draw coastlines, country boundaries, fill continents.

map.fillcontinents(color='white',lake_color='#85A6D9')

map.drawcoastlines(color='#6D5F47', linewidth=.4)

map.drawcountries(color='#6D5F47', linewidth=.4)



# lat/lon coordinates of top ten indian cities

lngs=array(ten_pop_cities['longitude'])

lats=array(ten_pop_cities['latitude'])

populations = array(ten_pop_cities['population_total'])

nc=array(ten_pop_cities['name_of_city'])



#scale populations to emphasise different relative pop sizes

s_populations = [p/5000 for p in populations]



# compute the native map projection coordinates for cities

x,y = map(lngs,lats)



#scatter scaled circles at the city locations

map.scatter(

    x,

    y,

    s=s_populations, #size

    c=s_populations, #color

    marker='o', #symbol

    alpha=0.75, #transparency

    zorder = 2, #plotting order

    cmap = cm.Dark2

    )



# plot population labels of the ten cities.

for name, xpt, ypt in zip(nc, x, y):

    plt.text(

        xpt+60000,

        ypt+30000,

        name,

        fontsize= 15,

        fontweight='bold',

        horizontalalignment='center',

        verticalalignment='center',

        zorder = 3,

        )

    

#add a title and display the map on screen

plt.title('Top Ten Cities of India By Population', fontsize=20)

plt.show()
def plot_map(sz, colorbarVal, title):

    plt.subplots(figsize=(20, 15))



    map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

    #set a background colour

    map.drawmapboundary(fill_color='#85A6D9')



    # draw coastlines, country boundaries, fill continents.

    map.fillcontinents(color='white',lake_color='#85A6D9')

    map.drawcoastlines(color='#6D5F47', linewidth=.4)

    map.drawcountries(color='#6D5F47', linewidth=.4)



    # lat/lon coordinates of top ten indian cities

    lngs=array(df['longitude'])

    lats=array(df['latitude'])



    # compute the native map projection coordinates for cities

    x,y = map(lngs,lats)



    #scatter scaled circles at the city locations

    cs = map.scatter(

        x,

        y,

        s=sz, #size

        c=sz, #color

        marker='o', #symbol

        alpha=0.5, #transparency

        zorder = 2, #plotting order

        cmap = cm.Dark2

        )

    

    # adding colorbar

    cbar = map.colorbar(cs, location='right',pad="5%")

    cbar.ax.set_yticklabels(colorbarVal)

    

    #add a title and display the map on screen

    plt.title(title, fontsize=20)

    plt.show()
plt.figure(figsize=(15, 5))

city_states = df.groupby(['state_name'])['name_of_city'].count().sort_values(ascending = False)

city_states.plot(kind = 'bar', fontsize = 12, )

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('States', fontsize=15)

plt.ylabel('Population', fontsize=15)

plt.title('States by number of cities in top 500', fontsize = 20)

plt.show ()
plt.figure(figsize=(15, 5))

pop_states = df[['population_total','state_name']].groupby(['state_name'])['population_total'].sum().sort_values(ascending = False)

pop_states.plot(kind = 'bar', fontsize = 12, alpha = 1)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('States', fontsize=15)

plt.ylabel('Population', fontsize=15)

plt.title('States by Total population in top 500', fontsize = 20)

plt.show ()
# Plotting the same on the map

population_sizes = df["population_total"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(df["population_total"].min(), df["population_total"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)

title = 'Population of Cities in Indian Map'

plot_map(population_sizes, colorbarValue, title)
plt.figure(figsize=(15, 5))

pop_male_states = df.groupby(['state_name'])['population_male'].sum().sort_values(ascending = False)

pop_male_states.plot(kind = 'bar', fontsize = 12)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('States', fontsize=15)

plt.ylabel('Population Male', fontsize=15)

plt.title('States by Total Male Population in Top 500 cities', fontsize = 20)

plt.show ()
# Plotting the same on the map

population_sizes = df["population_male"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(df["population_male"].min(), df["population_male"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)

title = "Male Population of Cities in India's Map"

plot_map(population_sizes, colorbarValue, title)
plt.figure(figsize=(15, 5))

pop_female_states = df.groupby(['state_name'])['population_female'].sum().sort_values(ascending = False)

pop_female_states.plot(kind = 'bar', fontsize = 12)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('Population Male', fontsize=15)

plt.ylabel('States', fontsize=15)

plt.title('States by most female populion in top 500 cities', fontsize = 20)

plt.show ()
# Plotting the same on the map

population_sizes = df["population_female"].apply(lambda x: int(x / 5000))

colorbarValue = np.linspace(df["population_female"].min(), df["population_female"].max(), 

                            num=10)

colorbarValue = colorbarValue.astype(int)

title = "Female Population of Cities in India's Map"

plot_map(population_sizes, colorbarValue, title)
male_pop_cities = df.sort_values(by = 'population_total', ascending = False)

ten_male_pop_cities = pop_cities.head(10)

ten_male_pop_cities
from mpl_toolkits.basemap import Basemap



plt.subplots(figsize=(20, 15))



map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

#set a background colour

map.drawmapboundary(fill_color='#85A6D9')



# draw coastlines, country boundaries, fill continents.

map.fillcontinents(color='white',lake_color='#85A6D9')

map.drawcoastlines(color='#6D5F47', linewidth=.4)

map.drawcountries(color='#6D5F47', linewidth=.4)



# lat/lon coordinates of top ten indian cities

lngs=array(ten_pop_cities['longitude'])

lats=array(ten_pop_cities['latitude'])

populations = array(ten_pop_cities['population_male'])

nc=array(ten_pop_cities['name_of_city'])



#scale populations to emphasise different relative pop sizes

s_populations = [p/5000 for p in populations]



# compute the native map projection coordinates for cities

x,y = map(lngs,lats)



#scatter scaled circles at the city locations

map.scatter(

    x,

    y,

    s=s_populations, #size

    c=s_populations, #color

    marker='o', #symbol

    alpha=0.75, #transparency

    zorder = 2, #plotting order

    cmap = cm.Dark2

    )



# plot population labels of the ten cities.

for name, xpt, ypt in zip(nc, x, y):

    plt.text(

        xpt+60000,

        ypt+30000,

        name,

        fontsize= 15,

        fontweight='bold',

        horizontalalignment='center',

        verticalalignment='center',

        zorder = 3,

        )

    

#add a title and display the map on screen

plt.title('Top Ten Cities of India By male population', fontsize=20)

plt.show()
male_pop_cities = df.sort_values(by = 'population_female', ascending = False)

ten_male_pop_cities = pop_cities.head(10)

ten_male_pop_cities
from mpl_toolkits.basemap import Basemap



plt.subplots(figsize=(20, 15))



map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

#set a background colour

map.drawmapboundary(fill_color='#85A6D9')



# draw coastlines, country boundaries, fill continents.

map.fillcontinents(color='white',lake_color='#85A6D9')

map.drawcoastlines(color='#6D5F47', linewidth=.4)

map.drawcountries(color='#6D5F47', linewidth=.4)



# lat/lon coordinates of top ten indian cities

lngs=array(ten_pop_cities['longitude'])

lats=array(ten_pop_cities['latitude'])

populations = array(ten_pop_cities['population_female'])

nc=array(ten_pop_cities['name_of_city'])



#scale populations to emphasise different relative pop sizes

s_populations = [p/5000 for p in populations]



# compute the native map projection coordinates for cities

x,y = map(lngs,lats)



#scatter scaled circles at the city locations

map.scatter(

    x,

    y,

    s=s_populations, #size

    c=s_populations, #color

    marker='o', #symbol

    alpha=0.75, #transparency

    zorder = 2, #plotting order

    cmap = cm.Dark2

    )



# plot population labels of the ten cities.

for name, xpt, ypt in zip(nc, x, y):

    plt.text(

        xpt+60000,

        ypt+30000,

        name,

        fontsize= 15,

        fontweight='bold',

        horizontalalignment='center',

        verticalalignment='center',

        zorder = 3,

        )

    

#add a title and display the map on screen

plt.title('Top Ten Cities of India By female population', fontsize=20)

plt.show()
lit_cities = df.sort_values('literates_total', ascending = False)

ten_lit_cities = lit_cities.head(10)

ten_lit_cities
from mpl_toolkits.basemap import Basemap



plt.subplots(figsize=(20, 15))



map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

#set a background colour

map.drawmapboundary(fill_color='#85A6D9')



# draw coastlines, country boundaries, fill continents.

map.fillcontinents(color='white',lake_color='#85A6D9')

map.drawcoastlines(color='#6D5F47', linewidth=.4)

map.drawcountries(color='#6D5F47', linewidth=.4)



# lat/lon coordinates of top ten indian cities

lngs=array(ten_lit_cities['longitude'])

lats=array(ten_lit_cities['latitude'])

populations = array(ten_lit_cities['literates_total'])

nc=array(ten_lit_cities['name_of_city'])



#scale populations to emphasise different relative pop sizes

s_populations = [p/5000 for p in populations]



# compute the native map projection coordinates for cities

x,y = map(lngs,lats)



#scatter scaled circles at the city locations

map.scatter(

    x,

    y,

    s=s_populations, #size

    c=s_populations, #color

    marker='o', #symbol

    alpha=0.75, #transparency

    zorder = 2, #plotting order

    cmap = cm.Dark2

    )



# plot population labels of the ten cities.

for name, xpt, ypt in zip(nc, x, y):

    plt.text(

        xpt+60000,

        ypt+30000,

        name,

        fontsize= 15,

        fontweight='bold',

        horizontalalignment='center',

        verticalalignment='center',

        zorder = 3,

        )

    

#add a title and display the map on screen

plt.title('Top Ten Cities of India by total literates', fontsize=20)

plt.show()
fig = plt.figure(figsize=(15,5))

lit_states = df.groupby(['state_name'])['effective_literacy_rate_total'].mean().sort_values(ascending = False)

lit_states.plot(kind = 'bar', fontsize = 12)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('Sates', fontsize = 15)

plt.ylabel('effective literates total', fontsize = 15)

plt.title('States by Average effective literates rate', fontsize = 20)

plt.show()
graduate_states  = df[["state_name","total_graduates","male_graduates","female_graduates"]].groupby("state_name").agg({"total_graduates":np.sum,

                                                                                                                        "male_graduates":np.sum,

                                                                                                                        "female_graduates":np.sum})



sort_graduates = graduate_states.sort_values(by=['total_graduates'], ascending=False)
sort_graduates.plot(kind = 'bar', figsize=(17,5), fontsize = 12, alpha = 1, colormap='Set2')

sort_graduates['total_graduates'].plot(kind='line',color = 'orange',linewidth=2.0, use_index = True)

plt.xticks(rotation = 90)

plt.xlabel('Sates', fontsize = 15)

plt.ylabel('number of Grduates in different states', fontsize = 15)

plt.title('States by number of total graduates in 500 cities', fontsize = 20)

plt.show()
df['graduate_ratio'] = (100 * df['total_graduates'] ) / df['population_total']
grad_rat_states = df.groupby(['state_name'])['graduate_ratio'].mean().sort_values(ascending = False)

fig = plt.figure(figsize=(15,5))

grad_rat_states.plot(kind = 'bar', fontsize = 12, alpha = 1, color='Orange')

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('Total Graduates', fontsize = 15)

plt.ylabel('Sates', fontsize = 15)

plt.title('States by total graduates percentage in entire population', fontsize = 20)

plt.show()
df['male_graduate_ratio'] = (100 * df['male_graduates'] ) / df['population_male']
grad_rat_states = df.groupby(['state_name'])['male_graduate_ratio'].mean().sort_values(ascending = False)

fig = plt.figure(figsize=(15,5))

grad_rat_states.plot(kind = 'bar', fontsize = 12, alpha = 1, color='Orange')

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('Total Male Graduates', fontsize = 15)

plt.ylabel('Sates', fontsize = 15)

plt.title('States by male graduates percentage in male population', fontsize = 20)

plt.show()
df['female_graduate_ratio'] = (100 *df['female_graduates'] ) / df['population_male']
grad_rat_states = df.groupby(['state_name'])['female_graduate_ratio'].mean().sort_values(ascending = False)



fig = plt.figure(figsize=(15,5))

grad_rat_states.plot(kind = 'bar', fontsize = 12, alpha = 1, color='Orange')

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('Total Female Graduates', fontsize = 15)

plt.ylabel('Sates', fontsize = 15)

plt.title('States by female graduates percentage in female population', fontsize = 20)

plt.show()
cities = df.sort_values(by = 'population_total', ascending = False)

fifty_pop_cities = cities.head(50)



plt.figure(figsize = (15,15))

ax = sns.barplot(data = fifty_pop_cities, y = 'name_of_city', x = 'population_total', palette = 'Paired')

ax.set(xlabel='Population in million', ylabel='Name of City')

sns.plt.title('Top 50 cities by population', fontsize = 20)

plt.show()
df['total_grad_per'] = (100*df['total_graduates'])/df['population_total']

cities = df.sort_values(by = 'total_grad_per', ascending = False)

fifty_cities = cities.head(50)



plt.figure(figsize = (15,15))

ax = sns.barplot(data = fifty_cities, y = 'name_of_city', x = 'total_grad_per', palette = 'Paired')

ax.set(xlabel='Population in million', ylabel='Name of City')

sns.plt.title('Total graduate percentage in entire population of top 50 cities', fontsize = 20)

plt.show()
plt.figure(figsize = (12,6))

top_female_grad = df.sort_values(by=['female_graduates'],ascending=False).head(15)

sns.barplot(x='name_of_city', y='female_graduates', data=top_female_grad, palette = 'viridis')

plt.xticks(rotation = 90)

plt.xlabel('City', fontsize =15)

plt.title('Top 15 cities where female gradutes lives', fontsize = 20)

fig = plt.figure(figsize=(15,5))

df['diff_grad'] = df['male_graduates'] - df['female_graduates']

grad_disc_states = df.groupby('state_name')['diff_grad'].mean()

grad_disc_states.plot(kind = 'bar', color='orange')

plt.title('Difference in average number of male and female graduates', fontsize = 20)

plt.xlabel('State', fontsize = 15)

plt.ylabel('Diff', fontsize = 15)

plt.show()
fig = plt.figure(figsize=(15,5))

df['diff_child'] = df['0-6_population_male'] - df['0-6_population_female']

child_disc_states = df.groupby('state_name')['diff_child'].mean()

child_disc_states.plot(kind = 'bar', color='orange')

plt.title('Difference in average number of male and female kids (Age 0-6)', fontsize = 20)

plt.xlabel('State', fontsize = 15)

plt.ylabel('Diff', fontsize = 15)

plt.show()
fig = plt.figure(figsize=(15,5))

df['diff_lit'] = df['literates_male'] - df['literates_female']

lit_disc_states = df.groupby('state_name')['diff_lit'].mean()

lit_disc_states.plot(kind = 'bar', color='orange')

plt.title('Difference in average number of male and female literates', fontsize = 20)

plt.xlabel('State', fontsize = 15)

plt.ylabel('Diff', fontsize = 15)

plt.show()
fig = plt.figure(figsize=(15,5))

df['diff_efec_lit'] = df['effective_literacy_rate_male'] - df['effective_literacy_rate_female']

efec_lit_disc_states = df.groupby('state_name')['diff_efec_lit'].mean()

efec_lit_disc_states.plot(kind = 'bar', color='orange')

plt.title('Difference in average effective literates of male and female ', fontsize = 20)

plt.xlabel('State', fontsize = 15)

plt.ylabel('Diff', fontsize = 15)

plt.show()
fig = plt.figure(figsize=(15,5))

sex_rat_states = df.groupby('state_name')['sex_ratio'].mean().sort_values(ascending=False)

sex_rat_states.plot(kind="bar", fontsize = 12, color = 'orange')

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.title('Sex ratio across states', fontsize = 20)

plt.show ()
states = df[['literates_male', 'literates_female', 

               'state_name']].groupby('state_name').sum().sort_values(['literates_male', 'literates_female'], ascending=False)

plt.figure(figsize=(15, 5))

sns.set_palette(sns.color_palette("muted"))

states['literates_male'].plot(kind = 'line', ls="--", label = 'male')

states['literates_female'].plot(kind = 'line', label = 'female')

plt.xticks(range(len(states.index)), list(states.index), rotation=90)

plt.xlabel('States')

plt.title('Male vs female literates across states', fontsize = 20)

plt.show()
states = df[['male_graduates', 'female_graduates', 

               'state_name']].groupby('state_name').sum().sort_values(['male_graduates', 'female_graduates'], ascending=False)

plt.figure(figsize=(15, 5))

sns.set_palette(sns.color_palette("muted"))

states['male_graduates'].plot(kind = 'line', ls="--", label = 'male')

states['female_graduates'].plot(kind = 'line', label = 'female')

plt.xticks(range(len(states.index)), list(states.index), rotation=90)

plt.xlabel('States')

plt.title('Male vs female graduates across states', fontsize = 20)

plt.show()
child_states  = df[['state_name', '0-6_population_total', '0-6_population_male', '0-6_population_female']].groupby('state_name').agg({'0-6_population_total':np.average,

                                                                                                 '0-6_population_male':np.average,

                                                                                                '0-6_population_female':np.average}).sort('0-6_population_total', ascending=False)

child_states.plot(kind='bar', figsize=(16,5), alpha = 0.7, colormap = 'Set1', width=0.6)

plt.title('most children lives region across sates', fontsize = 20)
print("Top States having better Total Literacy Rates, Sex-Ratio and Graduation Ratio are as follows:")

los = [lit_states.head(20),sex_rat_states.head(20),grad_rat_states.head(20)]

pd.concat(los, axis=1, join='inner')
fig = plt.figure(figsize=(12,6))

sns.regplot(x="sex_ratio", y='graduate_ratio', data=df)

plt.title('Linear relation between sex ration and graduate ratio', fontsize = 20)
poor_states = df[['population_total', 'literates_total', 'total_graduates', 'sex_ratio', 'graduate_ratio',

               'state_name']].groupby('state_name').sum().sort_values(['population_total', 'literates_total', 'total_graduates', 'sex_ratio', 'graduate_ratio'], ascending = True)

poor_states['sex_ratio'] = poor_states['sex_ratio']*50

poor_states['graduate_ratio'] = poor_states['graduate_ratio']*1000

poor_states.head(5).plot(kind='bar', figsize=(12,6), alpha = 0.7, colormap = 'viridis', width = 0.7)

plt.xticks(rotation=90, fontsize = 12)

plt.xlabel('States', fontsize = 15)

plt.title('Top 5 undeveloped states of India', fontsize = 20)

plt.show()
undeveloped_cities = df.sort_values(by = ['total_graduates', 'literates_total', 'sex_ratio', 'graduate_ratio'], ascending = False)

undeveloped_cities = pop_cities.tail(10)

undeveloped_cities