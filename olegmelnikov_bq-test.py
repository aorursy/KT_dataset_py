import numpy as np, pandas as pd; from subprocess import check_output; import bq_helper
# Input data files are available in the "../input/" directory.
#print(check_output(["ls", "../input"]).decode("utf8"))
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
#open_aq.list_tables()
open_aq.head("global_air_quality")
#us_cities = open_aq.query_to_pandas_safe(query)
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'µg/m³'
        """
open_aq.query_to_pandas_safe(query)

query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """
open_aq.query_to_pandas_safe(query)