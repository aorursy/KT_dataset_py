# Load packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

import folium

import folium.plugins as plugins



import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_rows =10

%matplotlib inline
# customized query helper function in Kaggle

import bq_helper



# Helper object

openaq = bq_helper.BigQueryHelper(active_project='bigquery-public-data',

                                 dataset_name='openaq')

# List of table

openaq.list_tables()
# Table Schema



openaq.table_schema('global_air_quality')
openaq.head('global_air_quality')
query = """ SELECT value,country, pollutant, 

extract(year from timestamp) as year,

extract(month from timestamp) as month,

extract(day from timestamp) as day,

date(timestamp) as date, unit

from `bigquery-public-data.openaq.global_air_quality`

"""



df1 = openaq.query_to_pandas(query)

df1
df1['unit'].value_counts()
query = """SELECT  unit,COUNT(unit) as `count`

        FROM `bigquery-public-data.openaq.global_air_quality`

        GROUP BY unit

        """

unit = openaq.query_to_pandas(query)



plt.style.use('bmh')

f, ax1 = plt.subplots(figsize = (14,5))



ax1.pie(x=unit['count'],labels=unit['unit'],shadow=True,autopct='%1.1f%%',explode=[0,0.1],\

       startangle=90,)

ax1.set_title('Distribution of measurement unit')

explode = np.arange(0,0.1)
df1['year'].value_counts()
## cols: country, value, year

## year < 2020 

## value > 0

## unit = 'µg/m³'



query = """ select country, round(avg(value)) as avg_value

            from `bigquery-public-data.openaq.global_air_quality`

            where unit ='µg/m³' and extract(year from timestamp) < 2020 

            group by country

            having avg_value > 0 

            order by avg_value desc"""



before = openaq.query_to_pandas(query)

before.head(10)
# stats

before['avg_value'].describe()
plt.style.use('bmh')

plt.figure(figsize = (20,10))

sns.barplot(before['country'], before['avg_value'], palette = 'magma')

plt.xticks(rotation  = 90)

plt.title('Average pollution of air by countries in unit $ug/m^3$.')

plt.ylabel('Average AQI in $ug/m^3$')
query = """ 

select city, value,extract( year from timestamp) as year, extract(month from timestamp) as month 

from `bigquery-public-data.openaq.global_air_quality`

where country = 'CL'and extract( year from timestamp) = 2019

"""



cl = openaq.query_to_pandas(query)

print(cl)

print(cl.describe())
query = """ select country, round(avg(value)) as avg_value

            from `bigquery-public-data.openaq.global_air_quality`

            where unit ='µg/m³' and extract(year from timestamp) < 2020

            and value > 0 AND value < 10000

            -- value > 100000 to be considered as outliers

            group by country

            order by avg_value DESC"""



before = openaq.query_to_pandas(query)

before
# PLOT

plt.figure(figsize = (20,10))

sns.barplot(before['country'], before['avg_value'], palette ='magma')

plt.xticks(rotation  = 90)

plt.title('Average pollution of air by countries till 2019.')

plt.ylabel('Average AQI in $ug/m^3$')
query = """ 

select distinct country, extract(year from timestamp) as year,round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

group by country, year

having year = 2020 and avg_value > 0

order by avg_value desc

"""



in_2020 = openaq.query_to_pandas(query)

in_2020


# PLOT

#plt.style.use('bmh')

plt.figure(figsize = (20,10))

sns.barplot(in_2020['country'], in_2020['avg_value'], palette ='magma')

plt.xticks(rotation  = 90)

plt.title('Average pollution of air by countries in 2020.')

plt.ylabel('Average AQI in $ug/m^3$')
## Trying to combine two different graphs with 



combined = pd.merge(before, in_2020, how = 'inner', on ='country')

combined.rename(columns = {'avg_value_x': 'till_2019', 'avg_value_y' : 'in_2020'}, inplace = True)

combined.drop('year', axis = 1, inplace =True)

print(combined[combined['country'] == 'IN'])

print(combined[combined['country'] == 'IT'])

print(combined[combined['country'] == 'AT'])

combined.sort_values(by = 'country', inplace = True)





fig, ax1 = plt.subplots(figsize = (12,10))

color = 'tab:green'

#barplot

ax1.set_title('Air Quality of different Countries(till_2019: Bar, 2020: Line)', fontsize = 20)

ax1.set_ylabel('AQI till 2019', fontsize =16)

ax1.set_xlabel('Countries', fontsize =16)

ax1 = sns.barplot(x = 'country', y = 'till_2019', data = combined, palette = 'summer')

ax1.tick_params(axis = 'y')



#lineplot

ax2 = ax1.twinx()

color ='tab:red'

ax2.set_ylabel('AQI in 2020', fontsize =16)

ax2 = sns.lineplot(x = 'country', y = 'in_2020', data = combined, color = color)

ax2.tick_params(axis = 'y', color =color)
## POLLUTANT IN TOP 5 POLLUTING COUNTRY till 2019

## used SUBQUERY to make FIG dynamic as the top 5 countries might change due to update of data.



query = """ 

select country, pollutant,round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality` 

where country in (select country from `bigquery-public-data.openaq.global_air_quality` 

                  where value > 0 and value < 10000 and unit  = 'µg/m³'and extract(year from timestamp) < 2020

                  group by country

                  order by avg(value) desc 

                  limit 5) and value > 0 and value < 10000 and unit  = 'µg/m³'

and extract(year from timestamp)  < 2020

group by country, pollutant

"""



top = openaq.query_to_pandas(query)

top

pivot = top.pivot(index = 'country',columns = 'pollutant', values = 'avg_value')

pivot = pivot.fillna(0)

pivot
pivot.plot.bar(stacked = True, color = ['red','green','blue','pink','orange','purple'], figsize = (10,7))

plt.xticks(rotation = 360)

plt.ylabel('Average Value of different pollutants till 2019')

plt.title('Distirbution of Pollutants in Top 5 Countries till 2019')
query = """ select country, pollutant, value

from `bigquery-public-data.openaq.global_air_quality`

where country = 'SG'"""



sg = openaq.query_to_pandas(query)

sg
## pollutant in 2020

## used SUBQUERY to make my FIG dynamic as the top 5 countries get changed.



query = """ 

select country, pollutant,round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality` 

where country in (select country from `bigquery-public-data.openaq.global_air_quality` 

                  where value > 0 and value < 10000 and unit  = 'µg/m³'and extract(year from timestamp)  = 2020

                  group by country

                  order by avg(value) desc 

                  limit 5) and value > 0 and value < 10000 and unit  = 'µg/m³'

                

and extract(year from timestamp)  = 2020

group by country, pollutant

"""



top_2020 = openaq.query_to_pandas(query)

#print(top_2020)





pivot_2020 = top_2020.pivot(index = 'country',columns = 'pollutant', values = 'avg_value')

pivot_2020 = pivot_2020.fillna(0)

#print(pivot_2020)



pivot_2020.plot.bar(stacked = True, color = ['red','green','blue','pink','orange','purple'], figsize = (10,7))

plt.xticks(rotation = 360)

plt.ylabel('Average Value of different pollutants in 2020')

plt.title('Distirbution of Pollutants in Top 5 Countries in 2020')
query = """ 

select country, pollutant,round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value >0 and value < 10000 and extract(year from timestamp) < 2020

group by country, pollutant

order by avg_value desc



"""



cor_before = openaq.query_to_pandas(query)

cor_before
# By country

p1_pivot = cor_before.pivot(index = 'country',values='avg_value', columns= 'pollutant')

p1_pivot = p1_pivot.fillna(0)

plt.figure(figsize=(14,15))

ax = sns.heatmap(p1_pivot, lw=0.01, cmap=sns.color_palette('Reds',500))

plt.yticks(rotation=30)

plt.title('Heatmap average AQI by Pollutant');
query = """ 

select country, pollutant,round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value >0 and value < 10000 and extract(year from timestamp) = 2020

group by country, pollutant

order by avg_value desc



"""



cor_2020 = openaq.query_to_pandas(query)

cor_2020
# By country

p2_pivot = cor_2020.pivot(index = 'country',values='avg_value', columns= 'pollutant')

p2_pivot = p2_pivot.fillna(0)

plt.figure(figsize=(10,15))

ax = sns.heatmap(p2_pivot, lw=1, cmap=sns.color_palette('Reds',500))

plt.yticks(rotation=20)

plt.title('Heatmap average AQI by Pollutant');
query = """ 

select extract(month from timestamp) as months, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value < 10000 and extract(year from timestamp) < 2020

group by months

order by months



"""





month =openaq.query_to_pandas(query)

month
plt.figure(figsize = (8,5))

sns.barplot(month['months'], month['avg_value'], palette = 'hls')

plt.title('Average value of AQI per month till 2019')
query = """ 

select extract(month from timestamp) as months, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value < 10000 and extract(year from timestamp) = 2020

group by months

order by months



"""





month_2020 =openaq.query_to_pandas(query)

month_2020
plt.figure(figsize = (8,5))

sns.barplot(month_2020['months'], month_2020['avg_value'])

plt.title('Average value of AQI per month in 2020')
## Before 2020



query = """ 



select country,city, latitude, longitude, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000 and extract(year from timestamp) < 2020

group by country, city, latitude, longitude

order by avg_value desc

"""



cities_before = openaq.query_to_pandas(query)

cities_before.dropna(inplace =True)

cities_before.head(10)
#Italy avg value



print('The average AQI for Italy till 2019 was {}'.format(round(cities_before[cities_before['country'] == 'IT']['avg_value'].mean(),2)))
## after



query = """ 



select country,city, avg(latitude) as latitudes, avg(longitude) as longitudes, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000 and extract(year from timestamp) = 2020

group by country, city

order by avg_value desc

"""



cities_after = openaq.query_to_pandas(query)

cities_after.dropna(inplace = True)

cities_after.head(10)



#Italy avg_value



print('The average AQI for Italy in 2020 is {}'.format(round(cities_before[cities_after['country'] == 'IT']['avg_value'].mean(),2)))
plt.style.use('ggplot')

f,ax =plt.subplots(figsize = (20,15))

m1 = Basemap(projection = 'cyl', llcrnrlon = -180, urcrnrlon =180, llcrnrlat =-90, urcrnrlat = 90,

            resolution = 'c', lat_ts= True)

m1.drawmapboundary(fill_color = '#A6CAE0', linewidth =0.2)

m1.fillcontinents(color ='grey', alpha =0.3)

m1.drawcountries(linewidth = 1, color = 'white')

m1.shadedrelief()



avg = np.log(cities_before['avg_value'])

m1loc =m1(cities_before['latitude'].tolist(), cities_before['longitude'])

m1.scatter(m1loc[1],m1loc[0],lw = 3, alpha =0.5, cmap ='hot_r', c =avg)

plt.title('Average AQI till 2019')

plt.colorbar(label=' Average Log AQI value in unit $ug/m^3$')

#Basemap?

plt.style.use('ggplot')

f,ax =plt.subplots(figsize = (20,15))

m1 = Basemap(projection = 'cyl', llcrnrlon = -180, urcrnrlon =180, llcrnrlat =-90, urcrnrlat = 90,

            resolution = 'c', lat_ts= True)

m1.drawmapboundary(fill_color = '#A6CAE0', linewidth =0.2)

m1.fillcontinents(color ='grey', alpha =0.3)

m1.drawcountries(linewidth = 1, color = 'white')

m1.shadedrelief()



avg = np.log(cities_after['avg_value'])

m1loc =m1(cities_after['latitudes'].tolist(), cities_after['longitudes'])

m1.scatter(m1loc[1],m1loc[0],lw = 3, alpha =0.5, cmap ='hot_r', c =avg)

plt.title('Average AQI in 2020')

plt.colorbar(label=' Average Log AQI value in unit $ug/m^3$')

#Basemap?

query = """ 

select extract(year from timestamp) as year, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000  and country  = 'IT' 

group by  year

order by year"""



italy = openaq.query_to_pandas(query)

italy
query = """ 

select pollutant, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000  and country  = 'IT' and extract(year from timestamp) < 2020

group by pollutant"""



it_pop_19 = openaq.query_to_pandas(query)

print('Pollutant till 2019')

print(it_pop_19)



print('\n')



query = """ 

select pollutant, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000  and country  = 'IT' and extract(year from timestamp) = 2020

group by pollutant"""



it_pop_20 = openaq.query_to_pandas(query)

print('Pollutant in 2020')

print(it_pop_20)
f, ax = plt.subplots(1,2, figsize = (14,5))

ax1, ax2 = ax.flatten()

ax1.pie(x = it_pop_19['avg_value'], labels = it_pop_19['pollutant'], shadow = True,

       autopct ='%1.0f%%', startangle = 90, colors = ['green','skyblue','orange'] )

ax1.set_title('Distribution of Pollutants in Italy till 2019')



ax2.set_title('Distribution of Pollutants in Italy in 2020')

ax2.pie(x = it_pop_20['avg_value'], labels = it_pop_20['pollutant'], shadow = True,

       autopct ='%1.0f%%', startangle = 80, pctdistance = 0.85, explode = (0,0,0,0,0.1,0))

centre_circle = plt.Circle((0,0),0.65,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax2.axis('equal')  

plt.tight_layout()

plt.show()





## Pollutant in BA (Bosnia And Herzegovina)



query = """ 

select pollutant, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000  and country  = 'BA' and extract(year from timestamp) < 2020

group by pollutant"""



ba_pop_19 = openaq.query_to_pandas(query)

print('Pollutant till 2019')

print(ba_pop_19)



print('\n')



query = """ 

select pollutant, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000  and country  = 'BA' and extract(year from timestamp) = 2020

group by pollutant"""



ba_pop_20 = openaq.query_to_pandas(query)

print('Pollutant in 2020')

print(ba_pop_20)
## Distribution of Pollutants in BA



f, ax = plt.subplots(1,2, figsize = (14,5))

ax1, ax2 = ax.flatten()

ax1.pie(x = ba_pop_19['avg_value'], labels = ba_pop_19['pollutant'], shadow = True,

       autopct ='%1.0f%%', startangle = 90)

ax1.set_title('Distribution of Pollutants in BA till 2019')



ax2.set_title('Distribution of Pollutants in BA in 2020')

ax2.pie(x = ba_pop_20['avg_value'], labels = ba_pop_20['pollutant'], shadow = True,

       autopct ='%1.0f%%', startangle = 40, pctdistance = 0.85, explode = (0,0.2,0,0,0))

centre_circle = plt.Circle((0,0),0.65,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax2.axis('equal')  

plt.tight_layout()

plt.show()
query = """



select extract(year from timestamp) as year, latitude, longitude, round(avg(value)) as avg_value

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000 and 

extract(year from timestamp) <= 2020

group by year, latitude, longitude

"""



p1 = openaq.query_to_pandas(query)

p1.sort_values(by = 'year', inplace = True)

p1





from matplotlib import animation,rc

import io

import base64

from IPython.display import HTML, display

import warnings

warnings.filterwarnings('ignore')

fig = plt.figure(figsize=(14,10))

plt.style.use('ggplot')



def animate(Year):

    ax = plt.axes()

    ax.clear()

    ax.set_title('Average AQI in Year: '+str(Year))

    m4 = Basemap(llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180,urcrnrlon=180,projection='cyl')

    m4.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

    m4.fillcontinents(color='grey', alpha=0.3)

    m4.drawcoastlines(linewidth=0.1, color="white")

    m4.shadedrelief()

    

    lat_y = list(p1[p1['year'] == Year]['latitude'])

    lon_y = list(p1[p1['year'] == Year]['longitude'])

    lat,lon = m4(lat_y,lon_y) 

    avg = p1[p1['year'] == Year]['avg_value']

    m4.scatter(lon,lat,c = avg,lw=2, alpha=0.3,cmap='hot_r')

    

   

ani = animation.FuncAnimation(fig,animate,list(p1['year'].unique()), interval = 1500)    

ani.save('animation.gif', writer='imagemagick', fps=1)

plt.close(1)

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
query = """

select pollutant, round(avg(value)) as avg_value_till_19

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000 and extract(year from timestamp) < 2020

group by pollutant"""



pop_19 = openaq.query_to_pandas(query)

pop_19
query = """

select pollutant, round(avg(value)) as avg_value_20

from `bigquery-public-data.openaq.global_air_quality`

where unit = 'µg/m³' and value > 0 and value <10000 and extract(year from timestamp) = 2020

group by pollutant"""



pop_20 = openaq.query_to_pandas(query)

pop_20
df= pd.merge(pop_19, pop_20, how ='inner', on ='pollutant')

df['perc_change'] = round((df['avg_value_till_19'] - df['avg_value_20'])*100/df['avg_value_till_19'])

df
df.plot.bar(x ='pollutant', stacked = True, figsize = (10,7))

plt.xticks(rotation = 360)