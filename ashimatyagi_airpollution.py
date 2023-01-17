# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_rows =10



import bq_helper



# google bigquery library for quering data

from google.cloud import bigquery



# BigQueryHelper for converting query result direct to dataframe

from bq_helper import BigQueryHelper
# matplotlib for plotting

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# import plotly

import plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.tools as tls

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as fig_fact

#plotly.tools.set_config_file(world_readable=True, sharing='public')



from mpl_toolkits.basemap import Basemap

import folium

import folium.plugins as plugins



%matplotlib inline
# object for the dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")

open_aq.list_tables()

#Schema 

open_aq.table_schema('global_air_quality')
# opening the dataset with few rows

df=open_aq.head("global_air_quality")

df
query = """

         SELECT Year, country, Average

                  FROM

                   (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,

                               AVG(value) as `Average`,country

                       FROM `bigquery-public-data.openaq.global_air_quality`

                      WHERE pollutant="pm25" AND unit='µg/m³' 

                      GROUP BY country, Year

                      ORDER BY Year )

                WHERE Year NOT IN (2020,2028)

        

        """

a= open_aq.query_to_pandas_safe(query)



a
query_aqi1 = """

         SELECT  avg(Average) as Avg, country

           FROM  (  SELECT Year, country, Average

                  FROM

                   (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,

                               AVG(value) as `Average`,country

                       FROM `bigquery-public-data.openaq.global_air_quality`

                      WHERE pollutant="pm25" AND unit='µg/m³' 

                      GROUP BY country, Year

                      ORDER BY Year )

                WHERE Year NOT IN (2020,2028) )

          GROUP BY country

            

        

        """

aqi1= open_aq.query_to_pandas_safe(query_aqi1)



aqi1
aqi1.sort_values(by=['Avg'], inplace=True, ascending=False)

aqi1=aqi1.head(30)

aqi1

query = """

    SELECT Year, value, pollutant, country, city

    FROM (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,value, unit, pollutant, country, city

    FROM `bigquery-public-data.openaq.global_air_quality`

    WHERE country = 'PL' and pollutant = 'pm25' and value > 0 )

    WHERE Year NOT IN (2028,2020)

"""

sg_pm25_dist = open_aq.query_to_pandas_safe(query)

sg_pm25_dist
query = """

    SELECT Year, value, pollutant, country, city

    FROM (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,value, unit, pollutant, country, city

    FROM `bigquery-public-data.openaq.global_air_quality`

    WHERE country = 'BR' and pollutant = 'pm25' and value > 0 )

    WHERE Year NOT IN (2028,2020)

"""

sg_pm25_dist = open_aq.query_to_pandas_safe(query)

sg_pm25_dist
query = """

    SELECT Year, value, pollutant, country, city

    FROM (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,value, unit, pollutant, country, city

    FROM `bigquery-public-data.openaq.global_air_quality`

    WHERE country = 'IN' and pollutant = 'pm25' and value > 0 )

    WHERE Year IN (2015)

"""

sg_pm25_dist = open_aq.query_to_pandas_safe(query)

sg_pm25_dist
query = """

    SELECT Year, value, pollutant, country, city

    FROM (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,value, unit, pollutant, country, city

    FROM `bigquery-public-data.openaq.global_air_quality`

    WHERE country = 'BE' and pollutant = 'pm25' and value > 0 )

    WHERE Year NOT IN (2028,2020)

"""

sg_pm25_dist = open_aq.query_to_pandas_safe(query)

sg_pm25_dist
query = """

    SELECT Year, value, pollutant, country, city

    FROM (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,value, unit, pollutant, country, city

    FROM `bigquery-public-data.openaq.global_air_quality`

    WHERE country = 'TR' and pollutant = 'pm25' and value > 0 )

    WHERE Year NOT IN (2028,2020)

"""

sg_pm25_dist = open_aq.query_to_pandas_safe(query)

sg_pm25_dist
aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="SG"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="PL"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="VN"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="MN"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="SE"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="PE"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="IL"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="MT"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="GH"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="GB"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="IE"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="DE"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="SK"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="CL"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="CZ"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="NO"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="PH"])

aqi1 = aqi1.drop(aqi1.index[aqi1['country'] =="SA"])

aq=aqi1.head(10)

aq
plt.style.use('bmh')

plt.figure(figsize=(10,5))

sns.barplot(aqi1['country'], aqi1['Avg'], palette='magma')

plt.xticks(rotation=45)

plt.ylabel("Average of PM2.5")

plt.title('Average PM2.5 Pollution by country(µg/m³)');
queryb = """

        

            (  SELECT Year, country, Average

                  FROM

                   (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,

                               AVG(value) as `Average`,country

                       FROM `bigquery-public-data.openaq.global_air_quality`

                      WHERE pollutant="pm25" AND unit='µg/m³' 

                      GROUP BY country, Year

                      ORDER BY Year )

                WHERE Year IN (2015) )

        

        

        """

b= open_aq.query_to_pandas_safe(queryb)

b = b.drop(b.index[b['country'] =="SG"])

b.sort_values(by=['Average'], inplace=True, ascending=False)

b

plt.style.use('bmh')

plt.figure(figsize=(10,5))

sns.barplot(b['country'], b['Average'], palette='magma')

plt.xticks(rotation=45)

plt.xlabel('Country')

plt.ylabel("Average of PM2.5")

plt.title('Average PM2.5 Pollution by country(µg/m³) in 2015');
queryc = """

        

            (  SELECT Year, country, Average

                  FROM

                   (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,

                               AVG(value) as `Average`,country

                       FROM `bigquery-public-data.openaq.global_air_quality`

                      WHERE pollutant="pm25" AND unit='µg/m³' 

                      GROUP BY country, Year

                      ORDER BY Year )

                WHERE Year IN (2016) )

        

        

        """

c= open_aq.query_to_pandas_safe(queryc)

c = c.drop(c.index[c['country'] =="SG"])

c = c.drop(c.index[c['country'] =="NL"])

c = c.drop(c.index[c['country'] =="CA"])

c = c.drop(c.index[c['country'] =="FR"])

c = c.drop(c.index[c['country'] =="DE"])

c.sort_values(by=['Average'], inplace=True, ascending=False)

c
plt.style.use('bmh')

plt.figure(figsize=(10,5))

sns.barplot(c['country'], c['Average'], palette='magma')

plt.xticks(rotation=45)

plt.xlabel('Country')

plt.ylabel("Average of PM2.5")

plt.title('Average PM2.5 Pollution by country(µg/m³) in 2016');
querye = """

        

            (  SELECT Year, country, Average

                  FROM

                   (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,

                               AVG(value) as `Average`,country

                       FROM `bigquery-public-data.openaq.global_air_quality`

                      WHERE pollutant="pm25" AND unit='µg/m³' 

                      GROUP BY country, Year

                      ORDER BY Year )

                WHERE Year IN (2018) )

        

        

        """

e= open_aq.query_to_pandas_safe(querye)

e = e.drop(e.index[e['country'] =="NL"])

e = e.drop(e.index[e['country'] =="SE"])

e = e.drop(e.index[e['country'] =="BE"])

e = e.drop(e.index[e['country'] =="PE"])

e = e.drop(e.index[e['country'] =="IE"])

e = e.drop(e.index[e['country'] =="NO"])

e = e.drop(e.index[e['country'] =="DE"])

e = e.drop(e.index[e['country'] =="PL"])

e = e.drop(e.index[e['country'] =="AU"])

e.sort_values(by=['Average'], inplace=True, ascending=False)

e
plt.style.use('bmh')

plt.figure(figsize=(10,5))

sns.barplot(e['country'], e['Average'], palette='magma')

plt.xticks(rotation=45)

plt.xlabel('Country')

plt.ylabel("Average of PM2.5")

plt.title('Average PM2.5 Pollution by country(µg/m³) in 2018');
queryf = """

        

            (  SELECT Year, country, Average

                  FROM

                   (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,

                               AVG(value) as `Average`,country

                       FROM `bigquery-public-data.openaq.global_air_quality`

                      WHERE pollutant="pm25" AND unit='µg/m³' 

                      GROUP BY country, Year

                      ORDER BY Year )

                WHERE Year IN (2019) )

        

        

        """

f= open_aq.query_to_pandas_safe(queryf)

f = f.drop(f.index[f['country'] =="NL"])

f = f.drop(f.index[f['country'] =="BE"])

f = f.drop(f.index[f['country'] =="MT"])

f = f.drop(f.index[f['country'] =="SA"])

f = f.drop(f.index[f['country'] =="DE"])

f = f.drop(f.index[f['country'] =="SK"])

f = f.drop(f.index[f['country'] =="CA"])

f = f.drop(f.index[f['country'] =="HR"])

f = f.drop(f.index[f['country'] =="PT"])

f = f.drop(f.index[f['country'] =="CZ"])

f = f.drop(f.index[f['country'] =="NO"])

f.sort_values(by=['Average'], inplace=True, ascending=False)

f
plt.style.use('bmh')

plt.figure(figsize=(10,5))

sns.barplot(f['country'], f['Average'], palette='magma')

plt.xticks(rotation=45)

plt.xlabel('Country')

plt.ylabel("Average of PM2.5")

plt.title('Average PM2.5 Pollution by country(µg/m³) in 2019');
q = """

    SELECT  country,  avg(Average) as Avg

    FROM (SELECT EXTRACT(YEAR FROM timestamp) as `Year`,avg(value) as Average, country

    FROM `bigquery-public-data.openaq.global_air_quality`

    WHERE   pollutant = 'pm10' and value > 0  and unit='µg/m³'

    GROUP BY country, Year   )

    WHERE Year IN (2015,2016,2017,2018,2019) and Average>35

    GROUP BY country

    ORDER BY Avg DESC

"""

s = open_aq.query_to_pandas_safe(q)

#s=s.head(10)

s
plt.style.use('bmh')

plt.figure(figsize=(15,5))

sns.barplot(s['country'], s['Avg'], palette='magma')

plt.xticks(rotation=45)

plt.xlabel('Country')

plt.ylabel("Average of PM10")

plt.title('Average PM10 Pollution by country(µg/m³) from 2015-2019');
query_aq= """

        SELECT EXTRACT(YEAR FROM timestamp) as `Year`, avg(value) as Average,  country

             FROM `bigquery-public-data.openaq.global_air_quality`

             WHERE country="IN" and pollutant="pm25" and value>0

             GROUP BY country, Year

            ORDER BY Year

        

         

        """

obj= open_aq.query_to_pandas_safe(query_aq)

obj.drop(obj.index[[5, 6]])
plt.plot(obj['Year'],obj['Average'])

plt.show()
#India PM2.5 average values from 2014-2018

query4 = """

        SELECT Year,Average

        FROM(

         SELECT EXTRACT(YEAR FROM timestamp) as `Year`,avg(value) as Average, country

                  FROM `bigquery-public-data.openaq.global_air_quality`

                       

                  WHERE pollutant = 'pm25' and value > 0 and country='IN'

                  GROUP BY  Year, country

                  ORDER BY Average DESC )

    

            

            ORDER BY Year 

            """

location = open_aq.query_to_pandas_safe(query4)

location=location.drop(location.index[[5, 6]])

location
#India PM10 average values from 2014-2018

query4 = """

        SELECT Year,Average

        FROM(

         SELECT EXTRACT(YEAR FROM timestamp) as `Year`,avg(value) as Average, country

                  FROM `bigquery-public-data.openaq.global_air_quality`

                       

                  WHERE pollutant = 'pm10' and value > 0 and country='IN'

                  GROUP BY  Year, country

                  ORDER BY Average DESC )

            

            ORDER BY Year 

            """



location1 = open_aq.query_to_pandas_safe(query4)

location1=location1.drop(location1.index[[4,5]])

location1
plt.style.use('bmh')

plt.figure(figsize=(10,5))

plt.plot(location['Year'], location['Average'], color='g', label='PM2.5')

plt.plot(location1['Year'], location1['Average'], color='orange', label='PM10')

plt.xticks(rotation=45)

plt.xlabel('Year')

plt.ylabel("Average")

plt.xlim(2013,2019)

plt.title('Average PM2.5 and PM10(µg/m³) in India from 2014 to 2018')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
#Indian cities PM10 average values from 2015-2019

query4 = """

         SELECT city, avg(Average) as Avg

        FROM ( 

         SELECT EXTRACT(YEAR FROM timestamp) as `Year`,avg(value) as Average, city, country

                  FROM `bigquery-public-data.openaq.global_air_quality`

                       

                  WHERE pollutant = 'pm10' and value > 0 and country="IN" and unit='µg/m³'

                  GROUP BY city, Year, country

                  ORDER BY Average DESC )

            WHERE Year  IN(2015,2016,2017,2018,2019) 

            GROUP BY city

        ORDER BY Avg DESC

            

            """

location = open_aq.query_to_pandas_safe(query4)

location = location.drop(location.index[location['city'] =="MUNICIPAL GUEST HOUSE COMPOUND"])

location.dropna(axis=0, inplace=True)

location



plt.style.use('bmh')

plt.figure(figsize=(15,5))

sns.barplot(location['city'], location['Avg'], palette='GnBu_d')

plt.xticks(rotation=45)

plt.xlabel('City')

plt.ylabel("Average of PM10")

plt.title('Average PM25 Pollution by city(µg/m³) from 2015 to 2019');
#Delhi PM2.5 average values from 2014-2018

query_d = """

        SELECT Year,Average, city, country

        FROM(

         SELECT EXTRACT(YEAR FROM timestamp) as `Year`,avg(value) as Average, city, country

                  FROM `bigquery-public-data.openaq.global_air_quality`

                       

                  WHERE pollutant = 'pm25' and value > 0 and city='Delhi'

                  GROUP BY  Year, city, country

                  ORDER BY Average DESC )

    

            

            ORDER BY Year 

            """

obj = open_aq.query_to_pandas_safe(query_d)

#obj=obj.drop(obj.index[[3,4]])

obj
#Delhi PM10 average values from 2014-2018

query_d = """

        SELECT Year,Average, city, country

        FROM(

         SELECT EXTRACT(YEAR FROM timestamp) as `Year`,avg(value) as Average, city, country

                  FROM `bigquery-public-data.openaq.global_air_quality`

                       

                  WHERE pollutant = 'pm10' and value > 0 and city='Delhi'

                  GROUP BY  Year, city, country

                  ORDER BY Average DESC )

    

            

            ORDER BY Year 

            """

obj1 = open_aq.query_to_pandas_safe(query_d)

obj1=obj1.drop(obj1.index[[3,4]])

obj1
plt.style.use('bmh')

plt.figure(figsize=(10,5))

plt.plot(obj['Year'],obj['Average'], color='purple', label='PM2.5')

plt.plot(obj1['Year'],obj1['Average'], color='yellow', label='PM10')

plt.xticks(rotation=45)

plt.xlabel('Year')

plt.ylabel("Average")

plt.xlim(2014,2019)

plt.title('Average PM2.5 and PM10(µg/m³) in Delhi from 2015 to 2018')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
#Locations in Delhi PM2.5 average values from 2015-2018

query_loc = """

        SELECT Year,Average, location,city, country

        FROM(

         SELECT EXTRACT(YEAR FROM timestamp) as `Year`,avg(value) as Average, location, city, country

                  FROM `bigquery-public-data.openaq.global_air_quality`

                       

                  WHERE pollutant = 'pm25' and value > 0 and city='Delhi'

                  GROUP BY  Year, city, country, location

                  ORDER BY Average DESC )

    

            WHERE Year IN(2017)

            ORDER BY Average DESC

            """

obj2 = open_aq.query_to_pandas_safe(query_loc)

#obj2=obj2.drop(obj2.index[[3,4]])

obj2
plt.style.use('bmh')

plt.figure(figsize=(15,5))

sns.barplot(obj2['location'], obj2['Average'], palette='cubehelix')

plt.xticks(rotation=45)

plt.xlabel('Location')

plt.ylabel("Average of PM2.5")

plt.title('Average PM2.5 Pollution by location(µg/m³) in Delhi for 2018')