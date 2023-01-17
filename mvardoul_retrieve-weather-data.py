import pandas as pd

from bq_helper import BigQueryHelper
noaa_dataset = BigQueryHelper(

        active_project= "bigquery-public-data", 

        dataset_name = "noaa_gsod"

    )
noaa_dataset.list_tables()
noaa_dataset.head('gsod2019', num_rows=10)

noaa_dataset.table_schema('gsod2019')
noaa_dataset.head('stations', num_rows=10)
query = """

    SELECT *

    FROM `bigquery-public-data.noaa_gsod.stations`

    WHERE country = "GR"

"""



GR_stations = noaa_dataset.query_to_pandas(query)

GR_stations


GR_stations[GR_stations['name'].str.contains('HERAKL')]
GR_stations[GR_stations['name'].str.contains('IRAKL')]
GR_stations[GR_stations['name'].str.contains('NIKOS')]
query = """

    SELECT *

    FROM `bigquery-public-data.noaa_gsod.gsod2000`

    WHERE stn = '167540'

    LIMIT 10

"""



noaa_dataset.query_to_pandas(query)
query = """

    SELECT year,mo,da,temp,dewp,visib,wdsp,prcp,fog,rain_drizzle,snow_ice_pellets,hail,thunder,tornado_funnel_cloud

    FROM `bigquery-public-data.noaa_gsod.gsod*`

    WHERE stn = '167540'

"""



noaa_dataset.estimate_query_size(query)

weather_data = noaa_dataset.query_to_pandas(query)

weather_data
weather_data_sorted = weather_data.sort_values(by=['year','mo','da']).reset_index().drop('index',axis=1)

weather_data_sorted.to_csv('dataset.csv', index=False)