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
# query to select all the countries where unit is not 'ppm'
query_ppm = """ select distinct country from `bigquery-public-data.openaq.global_air_quality` 
                where unit != 'ppm' """

# saving the query result to a dataframe countries
countries = open_aq.query_to_pandas_safe(query_ppm)

# print 5 distinct country names with unit != 'ppm'
countries.head()

# query to select pollutant where value = 0
query_value0 = """ select distinct pollutant from `bigquery-public-data.openaq.global_air_quality`
            where value = 0.00 """

# saving the query result to a data frame pollutants 
pollutants = open_aq.query_to_pandas_safe(query_value0)

# print 5 distinct pollutants with value = 0
pollutants.head()