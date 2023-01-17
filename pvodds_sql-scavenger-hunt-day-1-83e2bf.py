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
# query to select countries that use a unit other than ppm to measure any type of pollution? 
query2 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
              """

countries_nonppmunit = open_aq.query_to_pandas_safe(query2)
# What five cities have the most npn ppm as units?
countries_nonppmunit.country.value_counts().head()


#  query to select pollutant with value =0
query3 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value =0
        """
pollutants_zerovalue = open_aq.query_to_pandas_safe(query3)
# What top 5 pollutants with zero value?
pollutants_zerovalue.pollutant.value_counts().head()