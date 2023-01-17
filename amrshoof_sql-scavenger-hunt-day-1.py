import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

air_quality = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="openaq")
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

countries_not_using_ppm = air_quality.query_to_pandas_safe(query, max_gb_scanned=1)
countries_not_using_ppm.to_csv("countries_not_using_pp.csv")