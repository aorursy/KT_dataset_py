from google.cloud import bigquery

client = bigquery.Client()

dataset_ref = client.dataset('noaa_icoads', project='bigquery-public-data')

dset = client.get_dataset(dataset_ref)
[i.table_id for i in client.list_tables(dset)]
icoads_core_2017 = client.get_table(dset.table('icoads_core_2017'))

[i.name+", type: "+i.field_type for i in icoads_core_2017.schema]
schema_subset = [col for col in icoads_core_2017.schema if col.name in ('year', 'month', 'day', 'hour', 'latitude', 'longitude', 'sea_level_pressure', 'sea_surface_temp', 'present_weather')]

results = [x for x in client.list_rows(icoads_core_2017, start_index=100, selected_fields=schema_subset, max_results=10)]
for i in results:

    print(dict(i))
def estimate_gigabytes_scanned_h(query, bq_client):

    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun

    my_job_config = bigquery.job.QueryJobConfig()

    my_job_config.dry_run = True

    my_job = bq_client.query(query, job_config=my_job_config)

    BYTES_PER_GB = 2**30

    print("This query takes "+str(round(my_job.total_bytes_processed / BYTES_PER_GB, 2))+" GB of quota.")

    

def estimate_gigabytes_scanned(query, bq_client):

    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun

    my_job_config = bigquery.job.QueryJobConfig()

    my_job_config.dry_run = True

    my_job = bq_client.query(query, job_config=my_job_config)

    BYTES_PER_GB = 2**30

    return my_job.total_bytes_processed / BYTES_PER_GB
estimate_gigabytes_scanned_h("SELECT sea_level_pressure FROM `bigquery-public-data.noaa_icoads.icoads_core_1662_2000`", client)

estimate_gigabytes_scanned("SELECT sea_level_pressure FROM `bigquery-public-data.noaa_icoads.icoads_core_1662_2000`", client)
QUERY = """

        SELECT latitude, longitude, sea_surface_temp, wind_direction_true, amt_pressure_tend,  air_temperature, sea_level_pressure, wave_direction, wave_height, timestamp

        FROM `bigquery-public-data.noaa_icoads.icoads_core_2017`

        WHERE longitude > -74 AND longitude <= -44 AND latitude > 36 AND latitude <= 65 AND wind_direction_true <= 360

        """
estimate_gigabytes_scanned_h(QUERY, client)

import pandas as pd
df = client.query(QUERY).to_dataframe()
df.size
df.head(10)
print(df.latitude.size, df.longitude.size)
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np

# set up orthographic map projection with

# perspective of satellite looking down at 50N, 100W.

# use low resolution coastlines.

map = Basemap(projection='ortho',lat_0=45,lon_0=-60,resolution='l')

# draw coastlines, country boundaries, fill continents.

map.drawcoastlines(linewidth=0.25)

map.drawcountries(linewidth=0.25)

map.fillcontinents(color='coral',lake_color='aqua')

# draw the edge of the map projection region (the projection limb)

map.drawmapboundary(fill_color='aqua')

# draw lat/lon grid lines every 30 degrees.

map.drawmeridians(np.arange(0,360,30))

map.drawparallels(np.arange(-90,90,30))

# make up some data on a regular lat/lon grid.

lats = df['latitude'].values

lons = df['longitude'].values

x, y = map(lons, lats)

# contour data over the map.

cs = map.scatter(x,y)

plt.title('contour lines over filled continent background')

plt.show()
df_nans = df.isnull().sum(axis = 0)

print ("Number of NaN's by column:\n\n", df_nans, sep = "") # \n means newline, sep = "" removes space between elements of print command

# Now I want to know the percentage of NaN's in each column, so I need to get values of df_nans and divide by number of recordings in df. Let's look what type df_nans has

print ("\nType of df_nans:\n\n", type(df_nans), sep = "")

# After looking at the output of dir() command excluding built-in and private elements I understood that values of df_nans are accessable with .values element

# (uncomment next command to look at dir() output), also full dir() available if you want to look at built-in and private elements

# print ([f for f in dir(df_nans) if not f.startswith('_')]) # print (dir(df_nans))

# Let's divide df_nans by df row vount and output it with per cent sign

df_nans_perc = 100*df_nans/len(df.index)

pd.options.display.float_format = '{:,.2f}%'.format

print("\nPer cent of NaN's by column:\n\n", df_nans_perc, sep = "")

pd.options.display.float_format = '{:,.2f}'.format
#Let's create separate dataset with latitude, longitude, sea_surface_temp, wind_direction_true, air_temperature, sea_level_pressure, timestamp:

df_no_nans =df[['latitude', 'longitude', 'sea_surface_temp', 'wind_direction_true', 'air_temperature', 'sea_level_pressure', 'timestamp']]

#And now let's remove all rows with NaN elements:

print("Rows in df_no_nans:\nBefore dropping NaN's:", len(df_no_nans.index))

df_no_nans = df_no_nans.dropna()

print("After:                ", len(df_no_nans.index))
list(df_no_nans)
import seaborn as sns

sns.heatmap(df_no_nans.corr(), 

        xticklabels=df_no_nans.corr().columns,

        yticklabels=df_no_nans.corr().columns)