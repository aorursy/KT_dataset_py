# import package with helper functions 
import bq_helper
import pandas as pd

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
##Which countries use a unit other than ppm to measure any type of pollution? 
##(Hint: to get rows where the value *isn't* something, use "!=")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """


countries_not_ppm = open_aq.query_to_pandas_safe(query)
#Listing all unique countries 
countries_not_ppm_unique = countries_not_ppm.country.unique()

display(pd.DataFrame(countries_not_ppm_unique))
#Which pollutants have a value of exactly 0?
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            where value = 0
        """

pollutants_with_zero_value = open_aq.query_to_pandas_safe(query)
pollutants_with_zero_value_unique = pollutants_with_zero_value.pollutant.unique()

#display unique values only
pollutants_with_zero_value_unique = pollutants_with_zero_value.pollutant.unique()
pollutants_with_zero_value_unique_df = pd.DataFrame(pollutants_with_zero_value_unique)


display(pollutants_with_zero_value_unique_df)