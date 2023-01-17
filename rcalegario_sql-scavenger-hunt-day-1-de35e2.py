# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "country" column where the
# "unit" column is other than "ppm"
query1 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

country_no_ppm = open_aq.query_to_pandas_safe(query1)
#some countries that use units other then ppm
country_no_ppm.head()
# query to select all the items from the "pollutant" column where the
# "value" column is 0.00
query2 = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """

pollutants_0 = open_aq.query_to_pandas_safe(query2)
#pollutant that have value exactly 0
pollutants_0