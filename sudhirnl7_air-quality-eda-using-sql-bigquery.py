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

# Customized query helper function explosively in Kaggle
import bq_helper

# Helper object
openAQ = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='openaq')
# List of table
openAQ.list_tables()
#Schema 
openAQ.table_schema('global_air_quality')
openAQ.head('global_air_quality')
# Summary statics
query = """SELECT value,averaged_over_in_hours
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³'
            """
p1 = openAQ.query_to_pandas(query)
p1.describe()
query = """SELECT value,country 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value < 0
            """
p1 = openAQ.query_to_pandas(query)
p1.describe().T
query2 = """SELECT value,country,pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value > 0
            """
p2 = openAQ.query_to_pandas(query2)
print('0.99 Quantile',p2['value'].quantile(0.99))
p2.describe().T
p2[p2['value']>10000]
query = """SELECT country,COUNT(country) as `count`
    FROM `bigquery-public-data.openaq.global_air_quality`
    GROUP BY country
    HAVING COUNT(country) >10
    ORDER BY `count`
    """
cnt = openAQ.query_to_pandas_safe(query)
cnt.tail()

plt.style.use('bmh')
plt.figure(figsize=(14,4))
sns.barplot(cnt['country'], cnt['count'], palette='magma')
plt.xticks(rotation=45)
plt.title('Distribution of country listed in data');
#Average polution of air by countries
query = """SELECT AVG(value) as `Average`,country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY country
            ORDER BY Average DESC
            """
cnt = openAQ.query_to_pandas(query)
plt.figure(figsize=(14,4))
sns.barplot(cnt['country'],cnt['Average'], palette= sns.color_palette('gist_heat',len(cnt)))
plt.xticks(rotation=90)
plt.title('Average polution of air by countries in unit $ug/m^3$')
plt.ylabel('Average AQI in $ug/m^3$');
query = """SELECT city,latitude,longitude,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY latitude,city,longitude   
            """
location = openAQ.query_to_pandas_safe(query)
#Location AQI measurement center
m = folium.Map(location = [20,10],tiles='Mapbox Bright',zoom_start=2)

# add marker one by on map
for i in range(0,500):
    folium.Marker(location = [location.iloc[i]['latitude'],location.iloc[i]['longitude']],\
                 popup=location.iloc[i]['city']).add_to(m)
    
m #  DRAW MAP
query = """SELECT city,latitude,longitude,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY latitude,city,longitude   
            """
location = openAQ.query_to_pandas_safe(query)
location.dropna(axis=0, inplace=True)
plt.style.use('ggplot')
f,ax = plt.subplots(figsize=(14,10))
m1 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90, urcrnrlat=90,
            resolution='c',lat_ts=True)

m1.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m1.fillcontinents(color='grey', alpha=0.3)
m1.drawcoastlines(linewidth=0.1, color="white")
m1.shadedrelief()
m1.bluemarble(alpha=0.4)
avg = location['Average']
m1loc = m1(location['latitude'].tolist(),location['longitude'])
m1.scatter(m1loc[1],m1loc[0],lw=3,alpha=0.5,zorder=3,cmap='coolwarm', c=avg)
plt.title('Average Air qulity index in unit $ug/m^3$ value')
m1.colorbar(label=' Average AQI value in unit $ug/m^3$');
#USA location
query = """SELECT 
            MAX(latitude) as `max_lat`,
            MIN(latitude) as `min_lat`,
            MAX(longitude) as `max_lon`,
            MIN(longitude) as `min_lon`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US' """
us_loc = openAQ.query_to_pandas_safe(query)
us_loc
query = """ SELECT city,latitude,longitude,averaged_over_in_hours,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US' AND unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY latitude,city,longitude,averaged_over_in_hours,country """
us_aqi = openAQ.query_to_pandas_safe(query)
# USA
min_lat = us_loc['min_lat']
max_lat = us_loc['max_lat']
min_lon = us_loc['min_lon']
max_lon = us_loc['max_lon']

plt.figure(figsize=(14,8))
m2 = Basemap(projection='cyl', llcrnrlon=min_lon, urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat,
            resolution='c',lat_ts=True)
m2.drawcounties()
m2.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m2.fillcontinents(color='grey', alpha=0.3)
m2.drawcoastlines(linewidth=0.1, color="white")
m2.drawstates()
m2.bluemarble(alpha=0.4)
avg = (us_aqi['Average'])
m2loc = m2(us_aqi['latitude'].tolist(),us_aqi['longitude'])
m2.scatter(m2loc[1],m2loc[0],c = avg,lw=3,alpha=0.5,zorder=3,cmap='rainbow')
m1.colorbar(label = 'Average AQI value in unit $ug/m^3$')
plt.title('Average Air qulity index in unit $ug/m^3$ of US');
#INDIA location
query = """SELECT 
            MAX(latitude) as `max_lat`,
            MIN(latitude) as `min_lat`,
            MAX(longitude) as `max_lon`,
            MIN(longitude) as `min_lon`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'IN' """
in_loc = openAQ.query_to_pandas_safe(query)
in_loc
query = """ SELECT city,latitude,longitude,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'IN' AND unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY latitude,city,longitude,country """
in_aqi = openAQ.query_to_pandas_safe(query)
# INDIA
min_lat = in_loc['min_lat']-5
max_lat = in_loc['max_lat']+5
min_lon = in_loc['min_lon']-5
max_lon = in_loc['max_lon']+5

plt.figure(figsize=(14,8))
m3 = Basemap(projection='cyl', llcrnrlon=min_lon, urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat,
            resolution='c',lat_ts=True)
m3.drawcounties()
m3.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m3.fillcontinents(color='grey', alpha=0.3)
m3.drawcoastlines(linewidth=0.1, color="white")
m3.drawstates()
avg = in_aqi['Average']
m3loc = m3(in_aqi['latitude'].tolist(),in_aqi['longitude'])
m3.scatter(m3loc[1],m3loc[0],c = avg,alpha=0.5,zorder=5,cmap='rainbow')
m1.colorbar(label = 'Average AQI value in unit $ug/m^3$')
plt.title('Average Air qulity index in unit $ug/m^3$ of India');
# Unit query
query = """SELECT  unit,COUNT(unit) as `count`
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY unit
        """
unit = openAQ.query_to_pandas(query)
# Pollutant query
query = """SELECT  pollutant,COUNT(pollutant) as `count`
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY pollutant
        """
poll_count = openAQ.query_to_pandas_safe(query)
plt.style.use('fivethirtyeight')
plt.style.use('bmh')
f, ax = plt.subplots(1,2,figsize = (14,5))
ax1,ax2= ax.flatten()
ax1.pie(x=unit['count'],labels=unit['unit'],shadow=True,autopct='%1.1f%%',explode=[0,0.1],\
       colors=sns.color_palette('hot',2),startangle=90,)
ax1.set_title('Distribution of measurement unit')
explode = np.arange(0,0.1)
ax2.pie(x=poll_count['count'],labels=poll_count['pollutant'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('Set2',5),startangle=60,)
ax2.set_title('Distribution of pollutants in air');
query = """ SELECT pollutant,
                AVG(value) as `Average`,
                COUNT(value) as `Count`,
                MIN(value) as `Min`,
                MAX(value) as `Max`,
                SUM(value) as `Sum`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY pollutant
            """
cnt = openAQ.query_to_pandas_safe(query)
cnt 
query = """SELECT AVG(value) as`Average`,country, pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³'AND value BETWEEN 0 AND 10000
            GROUP BY country,pollutant"""
p1 = openAQ.query_to_pandas_safe(query)
# By country
p1_pivot = p1.pivot(index = 'country',values='Average', columns= 'pollutant')

plt.figure(figsize=(14,15))
ax = sns.heatmap(p1_pivot, lw=0.01, cmap=sns.color_palette('Reds',500))
plt.yticks(rotation=30)
plt.title('Heatmap average AQI by Pollutant');
f,ax = plt.subplots(figsize=(14,6))
sns.barplot(p1[p1['pollutant']=='co']['country'],p1[p1['pollutant']=='co']['Average'],)
plt.title('Co AQI in diffrent country')
plt.xticks(rotation=90);
f,ax = plt.subplots(figsize=(14,6))
sns.barplot(p1[p1['pollutant']=='pm25']['country'],p1[p1['pollutant']=='pm25']['Average'])
plt.title('pm25 AQI in diffrent country')
plt.xticks(rotation=90);
#source_name 
query = """ SELECT source_name, COUNT(source_name) as `count`
    FROM `bigquery-public-data.openaq.global_air_quality`
    GROUP BY source_name
    ORDER BY count DESC
    """
source_name = openAQ.query_to_pandas_safe(query)
plt.figure(figsize=(14,10))
sns.barplot(source_name['count'][:20], source_name['source_name'][:20],palette = sns.color_palette('YlOrBr'))
plt.title('Distribution of Top 20 source_name')
#plt.axvline(source_name['count'].median())
plt.xticks(rotation=90);
query = """SELECT averaged_over_in_hours, COUNT(*) as `count`
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY averaged_over_in_hours
            ORDER BY count DESC """
cnt = openAQ.query_to_pandas(query)
#cnt['averaged_over_in_hours'] = cnt['averaged_over_in_hours'].astype('category')
plt.figure(figsize=(14,5))
sns.barplot( cnt['averaged_over_in_hours'],cnt['count'], palette= sns.color_palette('brg'))
plt.title('Distibution of quality measurement per hour ');
query = """SELECT AVG(value) as`Average`,country,
            EXTRACT(YEAR FROM timestamp) as `Year`,
            pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm' 
            GROUP BY country,Year,pollutant"""
pol_aqi = openAQ.query_to_pandas_safe(query)
# By month in year
plt.figure(figsize=(14,8))
sns.barplot(pol_aqi['country'], pol_aqi['Average'])
plt.title('Distribution of average AQI by country $ppm$');
query = """SELECT EXTRACT(YEAR FROM timestamp) as `Year`,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY EXTRACT(YEAR FROM timestamp)
            """
quality = openAQ.query_to_pandas(query)

query = """SELECT EXTRACT(MONTH FROM timestamp) as `Month`,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY EXTRACT(MONTH FROM timestamp)
            """
quality1 = openAQ.query_to_pandas(query)
# plot
f,ax = plt.subplots(1,2, figsize= (14,6),sharey=True)
ax1,ax2 = ax.flatten()
sns.barplot(quality['Year'],quality['Average'],ax=ax1)
ax1.set_title('Distribution of average AQI by year')
sns.barplot(quality1['Month'],quality['Average'], ax=ax2 )
ax2.set_title('Distribution of average AQI by month')
ax2.set_ylabel('');
# by year & month
query = """SELECT EXTRACT(YEAR from timestamp) as `Year`,
            EXTRACT(MONTH FROM timestamp) as `Month`,
            AVG(value) as `Average`
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
        GROUP BY year,Month"""
aqi_year = openAQ.query_to_pandas_safe(query)
# By month in year
plt.figure(figsize=(14,8))
sns.pointplot(aqi_year['Month'],aqi_year['Average'],hue = aqi_year['Year'])
plt.title('Distribution of average AQI by month');
# Heatmap by country 
query = """SELECT AVG(value) as `Average`,
            EXTRACT(YEAR FROM timestamp) as `Year`,
            country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY country,Year
            """
coun_aqi = openAQ.query_to_pandas_safe(query)
coun_pivot = coun_aqi.pivot(index='country', columns='Year', values='Average').fillna(0)
# By month in year
plt.figure(figsize=(14,15))
sns.heatmap(coun_pivot, lw=0.01, cmap=sns.color_palette('Reds',len(coun_pivot)))
plt.yticks(rotation=30)
plt.title('Heatmap average AQI by YEAR');
query = """SELECT EXTRACT(YEAR FROM timestamp) as `Year`,AVG(value) as `Average`,
            latitude,longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
        GROUP BY Year, latitude,longitude
        """
p1 = openAQ.query_to_pandas_safe(query)
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
    
    lat_y = list(p1[p1['Year'] == Year]['latitude'])
    lon_y = list(p1[p1['Year'] == Year]['longitude'])
    lat,lon = m4(lat_y,lon_y) 
    avg = p1[p1['Year'] == Year]['Average']
    m4.scatter(lon,lat,c = avg,lw=2, alpha=0.3,cmap='hot_r')
    
   
ani = animation.FuncAnimation(fig,animate,list(p1['Year'].unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
# Continued