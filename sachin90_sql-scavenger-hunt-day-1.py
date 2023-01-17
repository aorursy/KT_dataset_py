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

# Query to select all counties in which unit is not ppm
query1= """ SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit!= 'ppm'
            ORDER BY country
        """

# Using query_to_pandas_safe to avoid possible usage greter than 1 GB
no_ppm_countries=open_aq.query_to_pandas_safe(query1)

# Display the countries having unit other than ppm
no_ppm_countries
# Query to select pollutants which have value 0 
query2= """ SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0
            ORDER BY pollutant
        """
# Getting the result in dataframe in safe mode
zero_pollutants=open_aq.query_to_pandas_safe(query2)

# Display query results
zero_pollutants
zero_pollutants