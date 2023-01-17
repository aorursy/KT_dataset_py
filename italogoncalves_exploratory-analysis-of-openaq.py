import bq_helper
import matplotlib.pyplot as plt

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query01 = """SELECT COUNT(location) AS measurements, city, country
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY city, country
            ORDER BY measurements DESC
"""

df01 = open_aq.query_to_pandas_safe(query01)

df01.head(10)
query02 = """SELECT DISTINCT(location), latitude, longitude
            FROM `bigquery-public-data.openaq.global_air_quality`
"""

df02 = open_aq.query_to_pandas_safe(query02)

plt.plot(df02["longitude"], df02["latitude"], ".")