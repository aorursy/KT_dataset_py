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
# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


# print all the tables in this dataset
# open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
# open_aq.head("global_air_quality")

# query to select all the items from the "country" column where the
# "unit" column is not "ppm"
query_for_first_question = """SELECT country
                              FROM `bigquery-public-data.openaq.global_air_quality`
                              WHERE unit != 'ppm'
                           """

# check the size of the query
# open_aq.estimate_query_size(query_for_first_question)

# run the query
countries = open_aq.query_to_pandas_safe(query_for_first_question)
# get top 5 countries name
countries.country.value_counts().head()

# query to select all the items from the "pollutant" column where the
# "value" column is "0.00"
query_for_second_question = """SELECT pollutant
                               FROM `bigquery-public-data.openaq.global_air_quality`
                               WHERE value = 0.00
                            """

# check the size of the query
# open_aq.estimate_query_size(query_for_second_question)

# run the query
pollutants = open_aq.query_to_pandas_safe(query_for_second_question)
# get top 5 pollutants name
pollutants.pollutant.value_counts().head()