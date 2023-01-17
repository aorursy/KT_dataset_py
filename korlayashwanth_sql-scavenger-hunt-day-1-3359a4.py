# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Importing bq helper library
import bq_helper

#creating an object out of helper library
bq_object = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',dataset_name = 'openaq')

#To get the list of all tables under openaq
# bq_object.list_tables()

#To get the schema of the table
# bq_object.table_schema('global_air_quality')

#To get the first few lines in a table
# bq_object.head('global_air_quality')

#query to get country whose units are not equal to ppm
query1 = """
        SELECT country FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != 'ppm'
        """

# function to get the size of the query
# bq_object.estimate_query_size(query1)

#Query 1 execution and this query fails when the scanned size is greater than 0.1 gb
country_without_unit_as_ppm = bq_object.query_to_pandas_safe(query1,max_gb_scanned = 0.1)
country_without_unit_as_ppm.to_csv("country_without_unit_as_ppm.csv")

#query to select pollutants whose value is equal to zero
query2 = """
        SELECT pollutant FROM `bigquery-public-data.openaq.global_air_quality` WHERE value = 0
        """
# bq_object.estimate_query_size(query2)

pollutant_value_zero = bq_object.query_to_pandas_safe(query2,max_gb_scanned = 0.1)
pollutant_value_zero.to_csv("pollutant_value_zero.csv")

#no of rows for query2
print(pollutant_value_zero.count())

#no of rows for query1
print(country_without_unit_as_ppm.count())