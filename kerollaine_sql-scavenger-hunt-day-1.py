import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='openaq')

open_aq.list_tables()
open_aq.head('global_air_quality')
query = """ SELECT city 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """

cities_us = open_aq.query_to_pandas_safe(query)
cities_us.city.value_counts().head()
query_question_one = """ SELECT DISTINCT(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
open_aq.estimate_query_size(query_question_one)
pollutant_not_ppm = open_aq.query_to_pandas_safe(query_question_one)
pollutant_not_ppm.head()
query_question_two = """ SELECT country, city, timestamp, pollutant 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
open_aq.estimate_query_size(query_question_two)
pollutant_equal_zero = open_aq.query_to_pandas_safe(query_question_two)
pollutant_equal_zero.head()
