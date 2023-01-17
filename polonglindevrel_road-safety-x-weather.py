# This Python 3 environment comes with many helpful analytics libraries installed



#import numpy as np

import pandas as pd

#import os



traffic = pd.read_csv("https://storage.googleapis.com/external-pl/Q2_Road_Safety_Data_2018_DataGov.csv")

print(traffic.shape)

traffic.head()
# List of column headers

traffic.dtypes
#install the convertbng package

!pip install convertbng



from convertbng.util import convert_lonlat



#create two new columns that convert Eastings/Northings into Longitudes/Latitudes

traffic['lon_accid'], traffic['lat_accid'] = convert_lonlat(traffic['Location_Easting_OSGR'], traffic['Location_Northing_OSGR'])



#show result



traffic[['Accident_Index','Location_Easting_OSGR', 'Location_Northing_OSGR',

        'lon_accid','lat_accid']].head()
# Set your own project id here

PROJECT_ID = 'my-us-project-244912'

  

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID, location="US")
dataset_id = 'trafficaccidents'



existing_datasets = list(client.list_datasets())



print('Existing datasets: {}'.format([ds.dataset_id for ds in existing_datasets]))
dataset_id = 'trafficaccidents'



existing_datasets = list(client.list_datasets())



if dataset_id in [ds.dataset_id for ds in existing_datasets]:

    print('Dataset "{}" already exists.'.format(dataset_id))

else:

    print('Creating new dataset as no existing dataset named "{}" found'.format(dataset))

    dataset = bigquery.Dataset(client.project+'.'+dataset_id)

    dataset.location = "US" #Specify the geographic location where the dataset should reside

    dataset = client.create_dataset(dataset_id)  # API request

    print("Created dataset {}.{}".format(client.project, dataset.dataset_id))
#@title Send All Weather Station Data to BigQuery Table

output_dataset_id = 'trafficaccidents' #@param{type:'string'}



output_table_id = 'accidents' #@param{type:'string'}



replace_or_append_output = 'replace' #@param{type:'string'}

  

job_config = bigquery.LoadJobConfig()



# Modify job config depending on if we want to replace or append to table

if(replace_or_append_output == 'replace'):

    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE  

else:  

    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND



dataset_ref = client.dataset(output_dataset_id)

table_ref = dataset_ref.table(output_table_id)



client.load_table_from_dataframe(

  dataframe = traffic,  ###

  destination = table_ref,

  job_config = job_config

  ).result()



# Optionally set explicit indices.

# If indices are not specified, a column will be created for the default

# indices created by pandas.



  

print('Accident Location output (' + replace_or_append_output + ') to ' + 

  client.project + ':' + output_dataset_id + '.' + output_table_id +

  '\n')
query = """

SELECT

  Accident_Index AS accident_id,

  DATETIME(PARSE_DATE('%d/%m/%Y',Date)) AS date,

  lat_accid,

  lon_accid

FROM

  `trafficaccidents.accidents`

ORDER BY

  accident_id

"""



query_job = client.query(

    query,

    # Location must match that of the dataset(s) referenced in the query.

    location="US",

)  # API request - starts the query



df = query_job.to_dataframe()

print(df.shape)

df.head(50)
query_stations = """

SELECT

  id AS stn, 

  name,

  latitude AS lat_stn, 

  longitude AS lon_stn

FROM

  `bigquery-public-data.ghcn_d.ghcnd_stations`

WHERE

  id LIKE 'UK%'

"""



query_job = client.query(

    query_stations,

    # Location must match that of the dataset(s) referenced in the query.

    location="US",

)  # API request - starts the query



df = query_job.to_dataframe()

print(df.shape)

df.head(10)
query = """

WITH

  ukstations AS (

    SELECT

      id AS stn, 

      name AS name_stn,

      latitude AS lat_stn, 

      longitude AS lon_stn

    FROM

      `bigquery-public-data.ghcn_d.ghcnd_stations`

    WHERE

      id LIKE 'UK%'

  ),

  stnacc AS (

  SELECT

    Accident_Index AS accident_id,

    DATETIME(PARSE_DATE('%d/%m/%Y',Date)) as date,

    lat_accid,

    lon_accid,

    stn,

    name_stn,

    lon_stn,

    lat_stn

  FROM

    `trafficaccidents.accidents`

  CROSS JOIN

    ukstations )

SELECT

  *

FROM (

  SELECT

    accident_id,

    date,

    lat_accid,

    lon_accid,

    stn,

    name_stn,

    lat_stn,

    lon_stn,

    dist_kms,

    ROW_NUMBER() OVER (PARTITION BY accident_id ORDER BY dist_kms) AS dist_rank

  FROM (

    SELECT

      *,

      ST_DISTANCE( ST_GEOGPOINT( lon_stn,

          lat_stn ),

        ST_GEOGPOINT( lon_accid,

          lat_accid )) / 1000 AS dist_kms

    FROM

      stnacc) )

WHERE

  dist_kms <= 60 # max dist of 60km from nearest stn

  AND dist_rank <= 3

"""



query_job = client.query(

    query,

    # Location must match that of the dataset(s) referenced in the query.

    location="US",

)  # API request - starts the query



df = query_job.to_dataframe()

print(df.shape)

df.head(10)
query = """

WITH

  ukstations AS (

    SELECT

      id AS stn, 

      name AS name_stn,

      latitude AS lat_stn, 

      longitude AS lon_stn

    FROM

      `bigquery-public-data.ghcn_d.ghcnd_stations`

    WHERE

      id LIKE 'UK%'

  ),

  stnacc AS (

  SELECT

    Accident_Index AS accident_id,

    DATETIME(PARSE_DATE('%d/%m/%Y',Date)) as date,

    lat_accid,

    lon_accid,

    stn,

    name_stn,

    lon_stn,

    lat_stn

  FROM

    `trafficaccidents.accidents`

  CROSS JOIN

    ukstations )

SELECT

  *

FROM (

  SELECT

    accident_id,

    date,

    lat_accid,

    lon_accid,

    stn,

    name_stn,

    lat_stn,

    lon_stn,

    dist_kms,

    ROW_NUMBER() OVER (PARTITION BY accident_id ORDER BY dist_kms) AS dist_rank

  FROM (

    SELECT

      *,

      ST_DISTANCE( ST_GEOGPOINT( lon_stn,

          lat_stn ),

        ST_GEOGPOINT( lon_accid,

          lat_accid )) / 1000 AS dist_kms

    FROM

      stnacc) )

WHERE

  dist_kms <= 60 # max dist of 60km from nearest stn

  AND dist_rank = 1

"""



query_job = client.query(

    query,

    # Location must match that of the dataset(s) referenced in the query.

    location="US",

)  # API request - starts the query



df = query_job.to_dataframe()

print(df.shape)

df.head(10)
# Retrieve data for the Heathrow station



stationid = 'UKM00003772' #LONDON HEATHROW AIRPORT weather station



query = """

SELECT

  id AS stn,

  date, #TIMESTAMP(PARSE_DATE('%Y-%m-%d', date)) as date,

  element,

  value

FROM

  `bigquery-public-data.ghcn_d.ghcnd_2018` AS wx

WHERE

  id = '{}'

  #AND qflag IS NULL

  AND element IN ('PRCP', 'TAVG')

ORDER BY

  date

LIMIT 10

""".format(stationid)



query_job = client.query(

    query,

    # Location must match that of the dataset(s) referenced in the query.

    location="US",

)  # API request - starts the query





df = query_job.to_dataframe()

df.head()
query = """

WITH

 ukstations AS (  # station data

  SELECT

    id AS stn,

    name as name_stn,

    latitude AS lat_stn,

    longitude AS lon_stn

  FROM 

    `bigquery-public-data.ghcn_d.ghcnd_stations`

  WHERE

    id LIKE 'UK%'

    ),

  

  #accident 

  

  acc AS ( # accidents data

  SELECT

    Accident_Index AS accident_id,

    DATE(DATETIME(PARSE_DATE('%d/%m/%Y',

          Date))) AS date,

    lat_accid,

    lon_accid,

    stn,

    name_stn,

    lon_stn,

    lat_stn

  FROM

    `trafficaccidents.accidents`

  CROSS JOIN

    ukstations 

    ),

  

  #combining accidents with nearest weather station

  

  accident_stations AS (

  SELECT

    *

  FROM (

    SELECT

      accident_id,

      date,

      lat_accid,

      lon_accid,

      stn,

      name_stn,

      lat_stn,

      lon_stn,

      dist_kms,

      ROW_NUMBER() OVER (PARTITION BY accident_id ORDER BY dist_kms) AS dist_rank

    FROM (

      SELECT

        *,

        ST_DISTANCE( ST_GEOGPOINT( lon_stn,

            lat_stn ),

          ST_GEOGPOINT( lon_accid,

            lat_accid )) / 1000 AS dist_kms

      FROM

        acc) )

  WHERE

    dist_kms <= 60 # max dist of 60km from nearest stn

    AND dist_rank = 1 #closest station

    ),

    

  # precipitation data based on date and weather station id

  precipitation AS (

  SELECT

    id AS stn,

    date,

    value AS prcp_mm

  FROM

    `bigquery-public-data.ghcn_d.ghcnd_2018`

  WHERE

    element = 'PRCP'),

    

  # temperature data based on date and weather station id

  tempavg AS (

  SELECT

    id AS stn,

    date,

    value AS tempavg_C

  FROM

    `bigquery-public-data.ghcn_d.ghcnd_2018`

  WHERE

    element = 'TAVG')

    

SELECT

  accident_id,

  date,

  lat_accid,

  lon_accid,

  stn,

  name_stn,

  lat_stn,

  lon_stn,

  dist_kms,

  tempavg_C,

  prcp_mm

FROM

  accident_stations

INNER JOIN

  precipitation #adding column: prcp_mm

USING

  (date,

    stn)

INNER JOIN

  tempavg #adding column: tempavg_C

USING

  (date,

    stn)

ORDER BY

  accident_id

"""



query_job = client.query(

    query,

    # Location must match that of the dataset(s) referenced in the query.

    location="US",

)  # API request - starts the query



stations = query_job.to_dataframe()

print(stations.shape)

stations.head(50) 
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def enable_plotly_in_cell():

  import IPython

  from plotly.offline import init_notebook_mode

  display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

  '''))

  init_notebook_mode(connected=False)
import plotly.plotly as py

import pandas as pd



def enable_plotly_in_cell():

  import IPython

  from plotly.offline import init_notebook_mode

  display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

  '''))

  init_notebook_mode(connected=False)





print(stations.head())

    

stations_dict = [ dict(

        type = 'scattergeo',

        locationmode = "country names",

        lon = stations['lon_stn'],

        lat = stations['lat_stn'],

        hoverinfo = 'text',

        text = stations['name_stn'],

        mode = 'markers',

        marker = dict(

            size=5,

            color='rgb(0, 0, 255)' #blue markers

        ))]



stations_nearest = [] #lines

for i in range( len( stations ) ):

    stations_nearest.append(

        dict(

            type = 'scattergeo',

            locationmode = "country names", 

            lon = [ stations['lon_accid'][i], stations['lon_stn'][i] ],

            lat = [ stations['lat_accid'][i], stations['lat_stn'][i] ],

            mode = 'lines',

            line = dict(

                width = 1,

                color = 'red',

            ),

            opacity = 0.5,

        )

    )



layout = dict(

        title = 'Traffic accidents to their nearest weather station (blue)',

        showlegend = False,

        autosize=False,

        width=1200,

        height=1200,

        geo = dict(

            #center=dict(lon=[-3],lat=[55]),

            scope="europe",

#            lataxis_range=[ 49,61],

#            lonaxis_range=[ -7, 2],

            lataxis_range=[stations['lat_accid'].min()-0.2, stations['lat_accid'].max()+0.2],

            lonaxis_range=[stations['lon_accid'].min()-0.2, stations['lon_accid'].max()+0.2],

            projection=dict( type='azimuthal equal area' ),

            showland = True,

            landcolor = 'rgb(243, 243, 243)',

            

            showsubunits = True,

            resolution = 50,

            

            showlakes = True,

            lakecolor = "rgb(255, 255, 255)",

            

            countrycolor = 'rgb(204, 204, 204)',

        ),

    )



enable_plotly_in_cell()



fig = dict( data=stations_nearest + stations_dict, layout=layout ) # + accid_dict

iplot( fig, filename='d3-traffic-accidents-weather' )