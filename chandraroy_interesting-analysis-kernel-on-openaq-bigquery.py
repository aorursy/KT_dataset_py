import bq_helper
air_quality = bq_helper.BigQueryHelper( active_project = "bigquery-public-data",
                                        dataset_name = "openaq")
# Look at the tables
air_quality.list_tables()
# Look at the schema of the table
air_quality.table_schema("global_air_quality")
# Extract name of all the countries in the openaq dataset
query = """ SELECT country, COUNT(country) as Measurements
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country
            ORDER BY Measurements
        """
country_list = air_quality.query_to_pandas_safe(query)
country_list.tail()
import matplotlib.pyplot as plt
import numpy as np
df = country_list
#y_pos = np.arrange(len(country_list.Measurements))
#df = np.sort( country_list['Measurements'])
plt.figure(figsize = (20, 20))
#plt.plot(df.country, df.Measurements, color="green")
plt.barh( df.country,df.Measurements, color="green")
plt.title("Country-wise Number of Measuring Points")

# Frequency of various units of measruements
query2 = """ SELECT unit, COUNT(unit) as Frequency
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY unit
            ORDER BY Frequency
        """
unit_list = air_quality.query_to_pandas_safe(query2)
unit_list.head()
plt.bar(unit_list.unit, unit_list.Frequency, color = 'blue')
air_quality.head("global_air_quality")
#Distribution of different kind of polutants
query3 = """ SELECT pollutant, AVG(value) as Value, COUNT(pollutant) as MeasurementPointsCount
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY pollutant
            ORDER BY Value 
            """

dist = air_quality.query_to_pandas_safe(query3)
dist.head()
import matplotlib.pyplot as plt
plt.figure(figsize= (10,5))
plt.bar(dist.pollutant, dist.Value, color = "red")
import matplotlib.pyplot as plt
plt.figure(figsize= (10,5))
plt.bar(dist.pollutant, dist.MeasurementPointsCount, color = "blue")
#Distribution of different kind of polutants
query4 = """ SELECT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant = 'bc'
            ORDER BY value 
            """
bc_values = air_quality.query_to_pandas_safe(query4)
bc_values.head()
air_quality.head("global_air_quality")
plt.plot(bc_values.value, color= "blue")
air_quality.head("global_air_quality")
#Distribution of different kind of polutants
query5 = """ SELECT EXTRACT(TIME FROM timestamp) as Date, AVG(Value) Vales
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant = 'co' and country = 'US'
            GROUP BY Date
            ORDER BY Date 
            """
df = air_quality.query_to_pandas_safe(query5)
df.head()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df.info()
#df['Date'] = pd.to_datetime(df.Date)
df.set_index('Date', inplace = True)
df.head()
df.shape
df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
import pandas as pd
time_bc_values['Date'] = pd.to_datetime(time_bc_values.Date)
time_bc_values.dtypes
air_quality.head("global_air_quality")
#plt.scatter(time_bc_values.Date, time_bc_values.Value_ag,  alpha=0.5)
#Distribution of different kind of polutants
query6 = """ SELECT value, pollutant, city, timestamp
           FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US' and pollutant = 'o3'
            """
df = air_quality.query_to_pandas_safe(query6)
df.head()
df.info()
df.shape
df.set_index('timestamp', inplace=True)
df.value.plot.box()
IQR = df.describe()[1]
df["value"].isnull().sum()
df.describe()
IQR = df.value.describe()[6]-df.value.describe()[4]
max_value = df.value.describe()[6] + (1.5 * IQR)
df.value[df.value > max_value] = max_value

df["value"].plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
query7 = """ SELECT country, COUNT(country) as No_stations
             FROM `bigquery-public-data.openaq.global_air_quality`
             GROUP BY country
             ORDER BY No_stations
             """
df = air_quality.query_to_pandas_safe(query7)
df.head()
df.tail()
# couont number of cities
query8 = """ SELECT city, COUNT(city) as No_stations
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE country = 'US'
             GROUP BY city
             ORDER BY No_stations
             """

df = air_quality.query_to_pandas_safe(query8)
df.head()
df.tail()
# couont number of cities
query9 = """ SELECT city, value, unit, pollutant, timestamp, latitude, longitude 
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE country = 'US'
             ORDER BY timestamp
             """
df = air_quality.query_to_pandas_safe(query9)
df.head()
df.tail()
df.loc[df.unit != 'ppm', 'value'] = df.value * 0.001
df.value.describe()
#df.set_index('timestamp', inplace=True)
df.head()
df.loc[df.pollutant == "o3","value"].plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
import seaborn as sns
#sns.barplot(x= "city", y="value", data =df, hue = "pollutant")
df.dtypes
df.head()
import datetime as dt
df['DD'] = df.timestamp.dt.day
df['MM'] = df.timestamp.dt.month
df['YYY'] = df.timestamp.dt.year
df['TIME'] = df.timestamp.dt.time
df['WEEKDAY'] = df.timestamp.dt.weekday
df.head()
df.loc[df.unit != 'ppm', 'value'] = df.value * 0.001
sns.barplot(x= "YYY", y="value", data =df, hue = "pollutant")
sns.barplot(x= "MM", y="value", data =df, hue = "pollutant")
plt.figure(figsize= (20, 20))
sns.barplot(x= "DD", y="value", data =df, hue = "pollutant")
sns.barplot(x= "TIME", y="value", data =df, hue = "pollutant")
plt.figure(figsize= (20, 10))
sns.boxplot(x= "WEEKDAY", y="value", data =df, hue = "pollutant")
f = sns.PairGrid(df, hue="pollutant")
f.map_diag(plt.hist)
f.map_offdiag(plt.scatter)
f.add_legend();
df.head()
# This part of code taken from http://geodesygina.com/matplotlib.html
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,\
            llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='i')
m.drawcoastlines()
m.drawcountries()
# draw parallels and meridians.
parallels = np.arange(-90.,91.,5.)
# Label the meridians and parallels
m.drawparallels(parallels,labels=[False,True,True,False])
# Draw Meridians and Labels
meridians = np.arange(-180.,181.,10.)
m.drawmeridians(meridians,labels=[True,False,False,True])
m.drawmapboundary(fill_color='white')
#plt.title("Forecast {0} days out".format(day_out))
long = df.loc[:,'longitude'].tolist()
lat = df.loc[:,'latitude'].tolist()
x,y = m(long, lat)                            # This is the step that transforms the data into the map's projection
m.plot(x,y, 'bo', markersize=5)
plt.show()