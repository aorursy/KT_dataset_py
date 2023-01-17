# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

query_countries = """SELECT DISTINCT country
                  FROM `bigquery-public-data.openaq.global_air_quality`
                  WHERE unit != 'ppm'
                  """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
non_ppm_countries = open_aq.query_to_pandas_safe(query_countries)

query_pollutants = """SELECT DISTINCT pollutant
                   FROM `bigquery-public-data.openaq.global_air_quality`
                   WHERE value = 0
                   """
zero_pollutants = open_aq.query_to_pandas_safe(query_pollutants)

#Save answer for question 1
non_ppm_countries.to_csv('countries.csv', index=False)
#Save answer for question 2
zero_pollutants.to_csv('pollutants.csv', index=False)