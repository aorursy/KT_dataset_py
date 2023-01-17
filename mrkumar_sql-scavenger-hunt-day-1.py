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
#solution 1 
query1="""select country from `bigquery-public-data.openaq.global_air_quality` where unit!='ppm' group by country 
"""
not_ppm_unit_country= open_aq.query_to_pandas_safe(query1)
not_ppm_unit_country.head()
not_ppm_unit_country.to_csv("country_unit_is_not_ppm.csv")

query2="""select pollutant from `bigquery-public-data.openaq.global_air_quality` where value=0.00"""
pollutant_value_zero=open_aq.query_to_pandas_safe(query2)
pollutant_value_zero.head()
pollutant_value_zero.to_csv("pollutant_zero.csv")