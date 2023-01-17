import numpy as np
import pandas as pd 
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
Query =  """select distinct(Country), unit 
from `bigquery-public-data.openaq.global_air_quality`
where unit != "ppm"
"""
us_country = open_aq.query_to_pandas_safe(Query)
us_country.Country.unique()
us_country_u = us_country.Country.unique()
us_country_U_df = pd.DataFrame(us_country_u)
display(us_country_U_df)
Query_1 =  """select Pollutant,Value
from `bigquery-public-data.openaq.global_air_quality`
where value = 0
"""
us_pollutant = open_aq.query_to_pandas_safe(Query_1)
us_pollutant_unique = us_pollutant.Pollutant.unique()
us_pollutant_unique_df = pd.DataFrame(us_pollutant_unique)
display(us_pollutant_unique_df)
