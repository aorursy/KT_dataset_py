# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# Your code goes here :)

query = """
    SELECT DISTINCT country, unit
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit != 'ppm'
    """
open_aq.estimate_query_size(query)
non_ppm_countries = open_aq.query_to_pandas_safe(query)
non_ppm_countries.head()
non_ppm_countries.describe()
non_ppm_countries
open_aq.table_schema("global_air_quality")
query = """
    SELECT DISTINCT pollutant
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE value = 0.0
    """
open_aq.estimate_query_size(query)
zero_pollutants = open_aq.query_to_pandas_safe(query)
zero_pollutants.describe()
zero_pollutants
query = """
    SELECT pollutant, city, country, timestamp
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE value = 0.0
    """
open_aq.estimate_query_size(query)
zero_pollutants = open_aq.query_to_pandas_safe(query)
zero_pollutants.describe()
zero_pollutants.head()