# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# countries that use a unit other than ppm
query = """SELECT DISTINCT country 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE pollutant != 'pm%'
         """
not_ppm = open_aq.query_to_pandas_safe(query)
not_ppm.country.head()
# pollutants having a value of exactly 0
query = """SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
         """
value_0 = open_aq.query_to_pandas_safe(query)
value_0.pollutant.head()