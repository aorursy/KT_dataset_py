# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

#use list function to retreive the list of all table
open_aq.list_tables()


open_aq.table_schema("global_air_quality")
# to obtain a sample of data we can use the head() function
open_aq.head("global_air_quality")
#let's write the 2 query 
query_unit = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
        """

query_value = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

#find our first result (locations using unit different from ppm) and print it heading the dataset
locations = open_aq.query_to_pandas_safe(query_unit)
locations.country.head()

# and the second
locations = open_aq.query_to_pandas_safe(query_value)
locations.pollutant.head()