import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality", 10)
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
non_ppm = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
open_aq.query_to_pandas_safe(non_ppm)
no_ppm = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
open_aq.query_to_pandas_safe(no_ppm)
