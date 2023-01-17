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

#* check how big this query will be
open_aq.estimate_query_size(query)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
#####
#* Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where
#*    the value *isn't* something, use "!=")
#* i. set up your query
query = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
      #* check how big this query will be
open_aq.estimate_query_size(query) #*(0.00019471719861030579 is GOOD)

#* ii. execute your query then()
nonppm_countries= open_aq.query_to_pandas(query)

#              #* you may view the first 7 rows
nonppm_countries.head(7)

# Your code goes here :)
#####
#* Which pollutants have a value of exactly 0?
#* i. set up your query
query = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
           #* check how big this query will be
open_aq.estimate_query_size(query) #*(0.00020727887749671936 is GOOD)

#* ii. execute your query then()
zero_pollut= open_aq.query_to_pandas(query)

#              #* you may view the first 7 rows
zero_pollut.head(7)