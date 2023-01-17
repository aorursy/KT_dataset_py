import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

import bq_helper 
Open_Air_Quality = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
Open_Air_Quality.list_tables()
Open_Air_Quality.table_schema("global_air_quality")
Open_Air_Quality.head("global_air_quality", num_rows = 5)
non_ppm_query = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
            """
Open_Air_Quality.estimate_query_size(non_ppm_query)
df_non_ppm = Open_Air_Quality.query_to_pandas_safe(non_ppm_query)
df_non_ppm ['country'].unique()
ppm_query = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = "ppm"
            """
df_ppm = Open_Air_Quality.query_to_pandas_safe(ppm_query)
df_ppm ['country'].unique()
Pol_0_query = """ SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            """
df_value_zero = Open_Air_Quality.query_to_pandas_safe(Pol_0_query)
df_value_zero ['pollutant'].unique()
