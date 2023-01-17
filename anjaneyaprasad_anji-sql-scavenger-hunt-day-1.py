import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="openaq")

open_aq.list_tables()
open_aq.head("global_air_quality")
query = """SELECT city
             FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country='US'
        """

us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
query = """SELECT pollutant 
             FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0
        """
pollutant_0 = open_aq.query_to_pandas_safe(query)
pollutant_0.head()
query = """SELECT unit
             FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

unit = open_aq.query_to_pandas_safe(query)
unit.head()