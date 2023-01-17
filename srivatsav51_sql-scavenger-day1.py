import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
open_aq.list_tables()
open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality")
q1="""SELECT city
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country='US'
"""
us_cities = open_aq.query_to_pandas_safe(q1)
us_cities.head()
us_cities.city.value_counts().head()
q2="""SELECT DISTINCT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit!='ppm'
"""
other_units = open_aq.query_to_pandas_safe(q2)
other_units.head()
q3="""SELECT DISTINCT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value=0.00
"""
zero_value_pollutants = open_aq.query_to_pandas_safe(q3)
zero_value_pollutants.head()