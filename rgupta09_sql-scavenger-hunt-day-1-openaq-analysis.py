# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# print information on all columns
open_aq.table_schema("global_air_quality")
# query to find Which countries use a unit other than ppm to measure any type of pollution
query = """SELECT DISTINCT(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# check how big this query will be
open_aq.estimate_query_size(query)
# Run the query in safe mode to get list of countries with non PPM measure
countries_non_ppm = open_aq.query_to_pandas_safe(query)
# List of countries where non PPM unit of measure has been used
countries_non_ppm
# query to verify the results
# Since US was in the list of countries that use unit other than ppm
# Verifying few rows for US where non PPM unit has been used
query1 = """SELECT location, city, country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' and country = 'US'
        """
# check how big this query will be
open_aq.estimate_query_size(query1)
# Run the query using safe mode
us_non_ppm = open_aq.query_to_pandas_safe(query1)
# Viewing rows for US where no PPM unit has been used
# Result shows that there is data from US where 'µg/m³' has been used as the unit of measure
# instead of PPM - this verifies the earlier output of list of countries
us_non_ppm.head()
# Saving list of countries which use units of measure other than PPM as a .csv 
countries_non_ppm.to_csv("countries_non_ppm.csv")
# query to find Which pollutants have a value of exactly 0
query_p = """SELECT DISTINCT(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# check how big this query will be
open_aq.estimate_query_size(query_p)
# Run the query using safe mode
pollutant_0 = open_aq.query_to_pandas_safe(query_p)
# Viewing list of pollutants with value 0
pollutant_0
# Verifying above list of pollutants actually contains value 0
query_p1 = """SELECT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 and pollutant = 'pm25'
        """
# check how big this query will be
open_aq.estimate_query_size(query_p1)
# Run the query using safe mode
pm25_0 = open_aq.query_to_pandas_safe(query_p1)
# Viewing  sample rows for pm25 where value = 0
# Result shows that there is data for pm25  where value = 0
# This verifies the earlier list of pollutants that came up for value 0
pm25_0.head()
# Saving list of pollutants with value 0 as a .csv 
pollutant_0.to_csv("pollutants_with_value_0.csv")