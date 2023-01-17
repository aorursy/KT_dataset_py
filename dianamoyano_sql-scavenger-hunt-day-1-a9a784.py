#Import package with helper functions
import bq_helper
#create a helper object for this dataset
open_aq= bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
Query1 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

#
country_noppm = open_aq.query_to_pandas_safe(Query1)
#
country_noppm

QueryP=""" SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY pollutant
HAVING sum(value)<0.001
"""
poll_zero=open_aq.query_to_pandas_safe(QueryP)
poll_zero