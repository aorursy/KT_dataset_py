import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()

open_aq.head("global_air_quality")
query = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit <> 'ppm'
        """
countries_without_ppm = open_aq.query_to_pandas_safe(query)
countries_without_ppm.country.value_counts()
query = """SELECT distinct pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

pollutant_with_zero = open_aq.query_to_pandas_safe(query)
pollutant_with_zero.pollutant.value_counts()






