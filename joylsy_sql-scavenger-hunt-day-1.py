# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# query to select all countries that uses units other than ppm
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
countries_not_ppm = open_aq.query_to_pandas_safe(query)

countries_not_ppm.country.value_counts()

# query to find which pollutants have a value of exactly 0?
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
pollutant_zerovalue = open_aq.query_to_pandas_safe(query)

pollutant_zerovalue.pollutant.value_counts()