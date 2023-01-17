import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query = """SELECT DISTINCT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != "ppm"
"""

No_ppm_countries = open_aq.query_to_pandas_safe(query)

No_ppm_countries.country.value_counts().head()
query_no_pollution = """SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""

No_polution_countries = open_aq.query_to_pandas_safe(query_no_pollution)
No_polution_countries.pollutant.value_counts().head()

