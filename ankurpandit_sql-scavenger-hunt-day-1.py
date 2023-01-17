# Your code goes here :)

import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# query to fetch countries that measure pollutant other than ppm
non_ppm_pollutants_query = """SELECT DISTINCT country
                        FROM `bigquery-public-data.openaq.global_air_quality`
                        WHERE pollutant != 'PM25' AND pollutant != 'PM10' 
                     """
non_ppm_countries = open_aq.query_to_pandas_safe(non_ppm_pollutants_query)
non_ppm_countries.to_csv("non_ppm_countries.csv")

# query to fetch pollutants which have value equal to zero
zero_value_pollutants_query = """SELECT DISTINCT pollutant
                                 FROM `bigquery-public-data.openaq.global_air_quality`
                                 WHERE value = 0
                              """
zero_value_pollutants = open_aq.query_to_pandas_safe(zero_value_pollutants_query)
zero_value_pollutants.to_csv("zero_value_pollutants.csv")
