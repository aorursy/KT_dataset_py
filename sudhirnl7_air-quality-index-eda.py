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

pd.options.display.max_rows =100
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.style.use('bmh')
# Customized query helper function explosively in Kaggle
import bq_helper

# Helper object
hist_aq = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='epa_historical_air_quality')
hist_aq.list_tables()
hist_aq.head('air_quality_annual_summary').head()
#poc
query = """SELECT COUNT(poc) as`count`,poc
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY poc"""
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
# datum
query = """SELECT COUNT(datum) as`count`,datum
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY datum"""
p2 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
# sample duration
query = """SELECT COUNT(sample_duration) as`count`,sample_duration
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY sample_duration"""
p3 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
# pollutant_standard
query = """SELECT COUNT(pollutant_standard) as`count`,pollutant_standard
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY pollutant_standard"""
p4 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
# event type
query = """SELECT COUNT(event_type) as`count`,event_type
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY event_type"""
p5 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)

# unit of measurement
query = """SELECT COUNT(units_of_measure) as`count`,units_of_measure
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY units_of_measure"""
p6 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
plt.style.use('bmh')
f, ax = plt.subplots(2,3,figsize = (14,10))
ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
ax1.pie(x=p1['count'],labels=p1['poc'],shadow=True,autopct='%1.1f%%',\
       colors=sns.color_palette('hot',2),startangle=60,)
ax1.set_title('Distribution of poc')

ax2.pie(x=p2['count'],labels=p2['datum'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('Set2',5),startangle=60,)
ax2.set_title('Distribution of datum')

ax3.pie(x=p3['count'],labels=p3['sample_duration'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('Set3',5),startangle=60,)
ax3.set_title('Distribution of sample dutration')

ax4.pie(x=p4['count'],labels=p4['pollutant_standard'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('inferno',5),startangle=60,)
ax4.set_title('Distribution of pollution standard')

ax5.pie(x=p5['count'],labels=p5['event_type'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('Set1',5),startangle=60,)
ax5.set_title('Distribution of event type')

ax6.pie(x=p6['count'],labels=p6['units_of_measure'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('hot',5),startangle=60,)
ax6.set_title('Distribution of units of measure');
# poc 
query = """SELECT COUNT(observation_count) as`count`,observation_count
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY observation_count
        ORDER BY observation_count DESC"""
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=1)
plt.figure(figsize= (14,4))
sns.barplot(p1['observation_count'], p1['count'], palette= sns.color_palette('tab10',len(p1)))

# aqi
query = """SELECT AVG(arithmetic_mean) as `Average`,
        EXTRACT (YEAR FROM date_local)as `year`
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY year
        ORDER BY year
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=1)
plt.figure(figsize=(14,4))
sns.barplot(p1['year'], p1['Average'], palette=sns.color_palette('hot',len(p1)))
plt.xticks(rotation=45);
# aqi
query = """SELECT aqi,
        EXTRACT(YEAR FROM date_local) as `Year`
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY Year,aqi
        ORDER BY Year
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=1)
plt.figure(figsize=(14,5))
sns.boxplot(x=p1['Year'],y =p1['aqi'], palette= sns.color_palette('winter',7))
plt.xticks(rotation=45)
plt.title('Distribution of AQI of CO by Year');
# aqi
query = """SELECT AVG(aqi) as `Average`,
        EXTRACT(YEAR FROM date_local) as `Year`,
        EXTRACT(MONTH FROM date_local) as `Month`
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY Year,Month
        ORDER BY Year
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=1)
plt.figure(figsize=(14,8))
sns.pointplot(x=p1['Year'],y =p1['Average'], hue=p1['Month'],\
              palette= 'gist_heat')
plt.xticks(rotation=45)
plt.title('Distribution of AQI of CO by Month');
# aqi
query = """SELECT AVG(aqi) as `Average`,state_name
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY state_name
        ORDER BY Average DESC
        LIMIT 20
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=2)
plt.figure(figsize=(14,8))
sns.barplot(p1['Average'], p1['state_name'], palette='viridis')
plt.title("Top 20 state's Average AQI");
query = """SELECT AVG(aqi) as `Average`,
        EXTRACT(YEAR FROM date_local) as `Year`,
        latitude,longitude,city_name
        FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
        GROUP BY city_name,latitude,longitude,Year
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=1)
hist_aq.estimate_query_size(query)
m = folium.Map(location= [40,-100],tiles='Mapbox Bright', zoom_start=4)

for i in range(0,2000):
    folium.Marker(location = [p1.iloc[i]['latitude'],p1.iloc[i]['longitude']],\
                 popup = p1.iloc[i]['city_name']).add_to(m)
m
from matplotlib import animation,rc
import io
import base64
from IPython.display import HTML, display

fig = plt.figure(figsize=(14,10))
plt.style.use('bmh')

def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('AQI in Year: '+str(Year))
    m1 = Basemap(llcrnrlat=15, urcrnrlat=70, llcrnrlon=-180, urcrnrlon= -60, \
                 projection='cyl', resolution='c')
    m1.drawcounties()
    m1.fillcontinents(color='grey', alpha=0.3)
    m1.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
    m1.drawcoastlines(linewidth=0.1, color="white")
    lat_year = p1[p1['Year'] == Year]['latitude']
    lon_year = p1[p1['Year'] == Year]['longitude']
    c_year = p1[p1['Year'] == Year]['Average']
    lat,lon = m1(lat_year,lon_year) 
    m1.scatter(lon,lat, c=c_year,\
               lw=2, alpha=0.3,cmap='inferno_r')
    #plt.clim(-1,20)
    

ani = animation.FuncAnimation(fig,animate,list(p1['Year'].unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
#plt.colorbar()
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

hist_aq.head('temperature_daily_summary').T
#poc
query = """SELECT COUNT(poc) as`count`,poc
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY poc"""
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
# datum
query = """SELECT COUNT(datum) as`count`,datum
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY datum"""
p2 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
# sample duration
query = """SELECT COUNT(sample_duration) as`count`,sample_duration
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY sample_duration"""
p3 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
# pollutant_standard
query = """SELECT COUNT(pollutant_standard) as`count`,pollutant_standard
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY pollutant_standard"""
p4 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
# event type
query = """SELECT COUNT(event_type) as`count`,event_type
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY event_type"""
p5 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)

# unit of measurement
query = """SELECT COUNT(units_of_measure) as`count`,units_of_measure
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY units_of_measure"""
p6 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=3)
plt.style.use('bmh')
f, ax = plt.subplots(2,3,figsize = (14,10))
ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
ax1.pie(x=p1['count'],labels=p1['poc'],shadow=True,autopct='%1.1f%%',\
       colors=sns.color_palette('hot',2),startangle=60,)
ax1.set_title('Distribution of poc')

ax2.pie(x=p2['count'],labels=p2['datum'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('Set2',5),startangle=60,)
ax2.set_title('Distribution of datum')

ax3.pie(x=p3['count'],labels=p3['sample_duration'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('Set3',5),startangle=60,)
ax3.set_title('Distribution of sample dutration')

ax4.pie(x=p4['count'],labels=p4['pollutant_standard'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('inferno',5),startangle=60,)
ax4.set_title('Distribution of pollution standard')

ax5.pie(x=p5['count'],labels=p5['event_type'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('Set1',5),startangle=60,)
ax5.set_title('Distribution of event type')

ax6.pie(x=p6['count'],labels=p6['units_of_measure'], shadow=True, autopct='%1.1f%%',\
        colors=sns.color_palette('hot',5),startangle=60,)
ax6.set_title('Distribution of units of measure');
# aqi
query = """SELECT AVG(arithmetic_mean) as `Average`,
        EXTRACT (YEAR FROM date_local)as `year`
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY year
        ORDER BY year
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=1)
plt.figure(figsize=(14,4))
sns.barplot(p1['year'], p1['Average'], palette=sns.color_palette('hot',len(p1)))
plt.xticks(rotation=45);
# aqi
query = """SELECT arithmetic_mean,
        EXTRACT(YEAR FROM date_local) as `Year`
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY Year,arithmetic_mean
        ORDER BY Year
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=1)
plt.figure(figsize=(14,5))
sns.boxplot(x=p1['Year'],y =p1['arithmetic_mean'], palette= sns.color_palette('winter',7))
plt.xticks(rotation=45)
plt.title('Distribution of AQI of CO by Year');
# aqi
query = """SELECT AVG(arithmetic_mean) as `Average`,
        EXTRACT(YEAR FROM date_local) as `Year`,
        EXTRACT(MONTH FROM date_local) as `Month`
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY Year,Month
        ORDER BY Year
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=1)
plt.figure(figsize=(14,8))
sns.pointplot(x=p1['Year'],y =p1['Average'], hue=p1['Month'],\
              palette= 'gist_heat')
plt.xticks(rotation=45)
plt.title('Distribution of temperature by Month');
# aqi
query = """SELECT AVG(arithmetic_mean) as `Average`,state_name
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY state_name
        ORDER BY Average DESC
        LIMIT 20
        """
p1 = hist_aq.query_to_pandas_safe(query,max_gb_scanned=2)
plt.figure(figsize=(14,8))
sns.barplot(p1['Average'], p1['state_name'], palette='gist_rainbow')
plt.title("Top 20 state's Average temperature");