# import package with helper functions 
import bq_helper as bq

# create a helper object for this dataset
open_aq = bq.BigQueryHelper(active_project="bigquery-public-data",
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
# query for getting a list of countries which use a unit other than ppm
query = """ select country from `bigquery-public-data.openaq.global_air_quality` where pollutant != 'ppm' """

non_ppm_countries = open_aq.query_to_pandas_safe(query)
non_ppm_countries.country       ## see list of countries 
non_ppm_countries.country.value_counts().head()    ## see top 5 countries
query = """ select pollutant from `bigquery-public-data.openaq.global_air_quality` where value=0 """
                                    
zero_pollutant = open_aq.query_to_pandas_safe(query)
zero_pollutant.pollutant     ## see list of pollutants
zero_pollutant.pollutant.value_counts().head()    ## see list of top 5