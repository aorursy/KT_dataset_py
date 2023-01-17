# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print information on all the columns in the "global_air_quality" table
# in the openaq dataset
open_aq.table_schema("global_air_quality")
# print the first couple rows of the "global_air_quality" table
open_aq.head("global_air_quality")


# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# check how big this query will be
open_aq.estimate_query_size(query)

# preview the first ten entries in the "city","country","pollutant" column of the global_air_quality table
open_aq.head("global_air_quality", selected_columns=("city","country","pollutant"), num_rows=10)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# lets describe the pandas dataframe
us_cities.describe()
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
#Question 1
query_unit = """SELECT distinct(country) 
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'ppm'
             """
countries_unit = open_aq.query_to_pandas_safe(query_unit)
countries_unit
#Question 2
query_pollutants = """SELECT distinct(pollutant)
                      FROM `bigquery-public-data.openaq.global_air_quality`
                      WHERE value = 0
                   """
pollutants_value = open_aq.query_to_pandas_safe(query_pollutants)
pollutants_value