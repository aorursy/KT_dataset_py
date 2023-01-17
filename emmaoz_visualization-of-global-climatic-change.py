import numpy as np

import pandas as pd

from bq_helper import BigQueryHelper



bq_assistant = BigQueryHelper(active_project= "bigquery-public-data", dataset_name= "noaa_gsod")
QUERY = """

        SELECT lat, lon

        FROM `bigquery-public-data.noaa_gsod.stations`

        """

bq_assistant.estimate_query_size(QUERY)
df = bq_assistant.query_to_pandas_safe(QUERY)

df.head()

#df.to_csv('noa_gsod_stations.csv')
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



m = Basemap(projection='cyl',lat_0=0, lon_0=-100, resolution='l', area_thresh=1000.0)

m.drawmapboundary(fill_color='aqua')

m.fillcontinents(color='coral',lake_color='aqua')

parallels = np.arange(-90.,90,30.)

m.drawparallels(parallels,labels=[False,True,True,False])

meridians = np.arange(0.,350.,30.)

m.drawmeridians(meridians,labels=[True,False,False,True])

lon = df['lon'].tolist()

lat = df['lat'].tolist()

xpt,ypt = m(lon,lat)

m.plot(xpt,ypt,'b+') 

plt.gcf().set_size_inches(18.5, 10.5)

plt.show()
QUERY = """

        SELECT stn, avg_temp, lat,lon

        FROM(

            SELECT stn,AVG(data.temp) AS avg_temp

            FROM `bigquery-public-data.noaa_gsod.gsod2017` AS data

            GROUP BY stn

        )temp_

        INNER JOIN `bigquery-public-data.noaa_gsod.stations`AS stations

        ON temp_.stn = stations.usaf

        

        """

bq_assistant.estimate_query_size(QUERY)
import time

t0 = time.time()

df = bq_assistant.query_to_pandas_safe(QUERY)

print(time.time()-t0)

print(df.head())

df['lat'].isnull().value_counts()
df = df.dropna(axis=0)

df.to_csv('noa_gsod_temp_2017.csv')

print('max_temp:%f'%df['avg_temp'].max())

print('min_temp:%f'%df['avg_temp'].min())
df_ = df.sample(5000)

m = Basemap(projection='cyl',lat_0=0, lon_0=-100, resolution='l', area_thresh=1000.0)

m.drawmapboundary()

m.drawcoastlines()

parallels = np.arange(-90.,90,30.)

m.drawparallels(parallels,labels=[False,True,True,False])

meridians = np.arange(0.,350.,30.)

m.drawmeridians(meridians,labels=[True,False,False,True])

lon = df_['lon'].tolist()

lat = df_['lat'].tolist()

df_['avg_temp'] = df_['avg_temp'].astype(int)

temp = df_['avg_temp'].tolist()

xpt,ypt = m(lon,lat)

cm = plt.cm.get_cmap('jet')

m.scatter(xpt, ypt, c=temp, cmap=cm, vmin = -70, vmax=100, s = 10, alpha = 0.4) 

plt.gcf().set_size_inches(18.5, 10.5)

plt.colorbar()

plt.show()
import time

t0 = time.time()

year_test_list = range(2009, 2018)

bq_assistant.max_wait_seconds = 3600

for year in year_test_list:

    query = '''

    SELECT stn, mo, avg_temp, avg_prcp, lat, lon

        FROM(

            SELECT stn, mo, AVG(data.temp) AS avg_temp, AVG(data.prcp) AS avg_prcp

            FROM `bigquery-public-data.noaa_gsod.gsod{0}` AS data

            GROUP BY stn, mo

        )temp_

        INNER JOIN `bigquery-public-data.noaa_gsod.stations`AS stations

        ON temp_.stn = stations.usaf

    '''.format(year)

    print(bq_assistant.estimate_query_size(query))

    qdata = bq_assistant.query_to_pandas_safe(query)

   # print(bq_assistant.max_wait_seconds)

    qdata.to_csv('noa_gsod_full_{0}.csv'.format(year))

    

print(time.time()-t0)