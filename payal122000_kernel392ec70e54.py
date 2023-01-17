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
from bq_helper import BigQueryHelper
noaa_dataset = BigQueryHelper(

        active_project= "bigquery-public-data", 

        dataset_name = "noaa_gsod"

    )
noaa_dataset.list_tables()
noaa_dataset.head('gsod2019', num_rows=10)
noaa_dataset.table_schema('gsod2019')
noaa_dataset.head('stations', num_rows=10)
noaa_dataset.table_schema('stations')
query = """

    SELECT *

    FROM `bigquery-public-data.noaa_gsod.stations`

    



"""



ALL_stations = noaa_dataset.query_to_pandas(query)

ALL_stations
query = """

    SELECT *

    FROM `bigquery-public-data.noaa_gsod.stations`

    WHERE country = "IN"

    



"""



IN_stations = noaa_dataset.query_to_pandas(query)

IN_stations
query = """

    SELECT *

    FROM `bigquery-public-data.noaa_gsod.gsod2019`

    WHERE stn='433330'

    

"""



noaa_dataset.query_to_pandas(query)
from google.cloud import bigquery

query = """

    SELECT year,mo,da,temp,dewp,visib,wdsp,prcp,fog,rain_drizzle,snow_ice_pellets,hail,thunder,tornado_funnel_cloud

    FROM (

  SELECT

    *

  FROM

    `bigquery-public-data.noaa_gsod.gsod2010` UNION ALL

  SELECT

    *

  FROM

    `bigquery-public-data.noaa_gsod.gsod2019` )

    WHERE stn='433330'

"""



noaa_dataset.estimate_query_size(query)
weather_data = noaa_dataset.query_to_pandas(query)

weather_data
weather_data_sorted = weather_data.sort_values(by=['year','mo','da']).reset_index().drop('index',axis=1)

weather_data_sorted.to_csv('portBlair.csv', index=False)