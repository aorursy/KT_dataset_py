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
not_ppm = """SELECT country,unit
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit != 'ppm'
          """

is_ppm = """SELECT country,unit
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit = 'ppm'
          """
country_not_ppm = open_aq.query_to_pandas_safe(not_ppm)
country_not_ppm.head()
country_is_ppm = open_aq.query_to_pandas_safe(is_ppm)
country_is_ppm.head()
no_pollutant = """SELECT location,pollutant,value
                  FROM `bigquery-public-data.openaq.global_air_quality`
                  WHERE value = 0.00
               """
not_pollutant = open_aq.query_to_pandas_safe(no_pollutant)
not_pollutant.head()
have_pollutant = """SELECT location,pollutant,value
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE value != 0.00
                 """
in_pollutant = open_aq.query_to_pandas_safe(have_pollutant)
in_pollutant.head()
# save our dataframe as a .csv 
country_not_ppm.to_csv("country_not_ppm.csv")
country_is_ppm.to_csv("country_is_ppm.csv")
not_pollutant.to_csv("not_pollutant.csv")
in_pollutant.to_csv("in_pollutant.csv")