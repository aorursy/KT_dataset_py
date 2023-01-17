# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_countries = open_aq.query_to_pandas_safe(query1)
us_countries.country.value_counts().head()
# Your code goes here :)
query2 = """SELECT pollutant, COUNT(*) as count
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
            GROUP BY pollutant
            ORDER BY count DESC
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_pollutants = open_aq.query_to_pandas_safe(query2)
us_pollutants.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

f, g = plt.subplots(figsize=(12, 9))
g = sns.barplot(x="pollutant", y="count", data=us_pollutants, palette="Blues_d")
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.title("Occurencies of Zero Value grouped by Pollutant")
plt.show(g)
# query to select coordinates of zero so2 sites
query_US = """SELECT  latitude, longitude, pollutant, value
                     FROM `bigquery-public-data.openaq.global_air_quality`
                     WHERE value = 0.0
                  """
US_value_coordinates = open_aq.query_to_pandas_safe(query_US)
US_value_coordinates.head()
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(30,15))
# miller projection
map = Basemap(projection='mill',lon_0=0)
# plot coastlines, draw label meridians and parallels.
map.drawcoastlines()
map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
map.drawmeridians(np.arange(map.lonmin,map.lonmax+30,60),labels=[0,0,0,1])
# fill continents 'coral' (with zorder=0), color wet areas 'aqua'
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')

colors = {
    'so2': 'black',
    'o3': 'red',
    'no2': 'blue',
    'co': 'yellow',
    'pm25': 'magenta',
    'pm10': 'white',
    'bc': 'green'
}

for index, row in US_value_coordinates.iterrows():
    x1,y1=map(row['longitude'],row['latitude'])  
    map.scatter(x1,y1,s=80,c=colors[row['pollutant']],marker="o",alpha=0.7,zorder=10)
    
plt.show()   

