import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import bq_helper
from mpl_toolkits.basemap import Basemap

pd.set_option('display.max_columns', 350)
pd.set_option('display.max_rows', 500)
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head("accident_2015")
query_drunk_2015 = """SELECT COUNT(consecutive_number) as Number_of_accidents,
                     SUM(number_of_drunk_drivers) as Drunk_drivers,
                     state_name
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                     WHERE number_of_drunk_drivers>0
                     GROUP BY state_name
                     ORDER BY Number_of_accidents DESC, Drunk_drivers DESC
                  """
drunk_2015 = accidents.query_to_pandas_safe(query_drunk_2015)
drunk_2015
query_drunk_2016 = """SELECT COUNT(consecutive_number) as Number_of_accidents,
                     SUM(number_of_drunk_drivers) as Drunk_drivers,
                     state_name
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers>0
                     GROUP BY state_name
                     ORDER BY Number_of_accidents DESC, Drunk_drivers DESC
                  """
drunk_2016 = accidents.query_to_pandas_safe(query_drunk_2016)
drunk_2016
plt.figure(figsize=(10,10))
gr_2015 = sns.barplot(x="Drunk_drivers", y="state_name", data=drunk_2015)
plt.title("Number of accidents per state in 2015")
plt.figure(figsize=(10,10))
gr_2016 = sns.barplot(x="Drunk_drivers", y="state_name", data=drunk_2016)
plt.title("Number of accidents per state in 2016")
query_drunk_perc_2015 = """SELECT 
                       ((SELECT count(*)*100
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                        WHERE number_of_drunk_drivers>0)
                        /
                        (SELECT count(*)
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`)) as Percentage
              """
drunk_percentage_2015 = accidents.query_to_pandas_safe(query_drunk_perc_2015)
drunk_percentage_2015
query_drunk_perc_2016 = """SELECT 
                       ((SELECT count(*)*100
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                        WHERE number_of_drunk_drivers>0)
                        /
                        (SELECT count(*)
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`)) as Percentage
              """
drunk_percentage_2016 = accidents.query_to_pandas_safe(query_drunk_perc_2016)
drunk_percentage_2016
query_drunk_hour_2015 = """SELECT COUNT(consecutive_number) as Number_of_accidents, 
                     EXTRACT(HOUR FROM timestamp_of_crash) as Hour
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                     WHERE number_of_drunk_drivers>0
                     GROUP BY Hour
                     ORDER BY Number_of_accidents DESC
                  """
accidents_drunk_hour_2015 = accidents.query_to_pandas_safe(query_drunk_hour_2015)
print(accidents_drunk_hour_2015)
sns.barplot(accidents_drunk_hour_2015.Hour, accidents_drunk_hour_2015.Number_of_accidents)
plt.title("Number of accidents per hour in 2015")
query_drunk_hour_2016 = """SELECT COUNT(consecutive_number) as Number_of_accidents, 
                     EXTRACT(HOUR FROM timestamp_of_crash) as Hour
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers>0
                     GROUP BY Hour
                     ORDER BY Number_of_accidents DESC
                  """
accidents_drunk_hour_2016 = accidents.query_to_pandas_safe(query_drunk_hour_2016)
print(accidents_drunk_hour_2016)
sns.barplot(accidents_drunk_hour_2016.Hour, accidents_drunk_hour_2016.Number_of_accidents)
plt.title("Number of accidents per hour in 2016")
query_weekday_2015 = """SELECT COUNT(consecutive_number) as Number_of_accidents, 
                     day_of_week
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                     WHERE number_of_drunk_drivers>0
                     GROUP BY day_of_week
                     ORDER BY Number_of_accidents DESC
                  """
weekday_2015 = accidents.query_to_pandas_safe(query_weekday_2015)
weekday_2015
sns.barplot(weekday_2015.day_of_week, weekday_2015.Number_of_accidents)
plt.title("Number of accidents per weekday in 2015")
query_weekday_2016 = """SELECT COUNT(consecutive_number) as Number_of_accidents, 
                     day_of_week
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     WHERE number_of_drunk_drivers>0
                     GROUP BY day_of_week
                     ORDER BY Number_of_accidents DESC
                  """
weekday_2016 = accidents.query_to_pandas_safe(query_weekday_2016)
weekday_2016
sns.barplot(weekday_2016.day_of_week, weekday_2016.Number_of_accidents)
plt.title("Number of accidents per state in 2016")
query_fatality_2015 = """SELECT number_of_fatalities,
                         number_of_drunk_drivers
                         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                      """
fatality_2015 = accidents.query_to_pandas_safe(query_fatality_2015)
fatality_2015.corr()
query_fatality_2016 = """SELECT number_of_fatalities,
                         number_of_drunk_drivers
                         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                      """
fatality_2016 = accidents.query_to_pandas_safe(query_fatality_2016)
fatality_2016.corr()
query_map_2015 = """SELECT latitude,
                          longitude
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                          where number_of_drunk_drivers>0
                      """
map_2015 = accidents.query_to_pandas_safe(query_map_2015)
map_2015.head()
plt.figure(figsize=(25,25))
plt.title("Map of accidents in US in 2015")
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
m.drawmapboundary(fill_color='cyan')
m.fillcontinents(color='coral',lake_color='cyan')
parallels = np.arange(0.,81,10.)
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])
xpt,ypt = m(np.array(map_2015.longitude),np.array(map_2015.latitude))
lonpt, latpt = m(xpt,ypt,inverse=True)
m.plot(xpt,ypt,'bp')
plt.show()
query_map_2016 = """SELECT latitude,
                          longitude
                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                          where number_of_drunk_drivers>0
                      """
map_2016 = accidents.query_to_pandas_safe(query_map_2016)
map_2016.head()
plt.figure(figsize=(25,25))
plt.title("Map of accidents in US in 2015")
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
m.drawmapboundary(fill_color='cyan')
m.fillcontinents(color='coral',lake_color='cyan')
parallels = np.arange(0.,81,10.)
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])
xpt,ypt = m(np.array(map_2016.longitude),np.array(map_2016.latitude))
lonpt, latpt = m(xpt,ypt,inverse=True)
m.plot(xpt,ypt,'bp')
plt.show()
query_land_use_2015 = """SELECT COUNT(consecutive_number) as Number_of_accidents,
                         land_use_name as Type_of_land
                         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                         WHERE number_of_drunk_drivers>0
                         GROUP BY land_use_name
                         ORDER BY Number_of_Accidents DESC
                      """
land_use_2015 = accidents.query_to_pandas_safe(query_land_use_2015)
land_use_2015
sns.barplot(land_use_2015.Type_of_land, land_use_2015.Number_of_accidents)
plt.title("Number of accidents per land type in 2015")
query_land_use_2016 = """SELECT COUNT(consecutive_number) as Number_of_accidents,
                         land_use_name as Type_of_land
                         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                         WHERE number_of_drunk_drivers>0
                         GROUP BY land_use_name
                         ORDER BY Number_of_Accidents DESC
                      """
land_use_2016 = accidents.query_to_pandas_safe(query_land_use_2016)
land_use_2016
sns.barplot(land_use_2016.Type_of_land, land_use_2016.Number_of_accidents)
plt.title("Number of accidents per land type in 2016")