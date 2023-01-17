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
# Query for question 1
query2="""SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """
# run query safely
countries_units_not_ppm = open_aq.query_to_pandas_safe(query2)

# view the 5 first rows of the returned dataframe
countries_units_not_ppm.head()
# save dataframe as .csv
countries_units_not_ppm.to_csv("countries_units_not_ppm.csv")
# Query for question 2
query3="""SELECT DISTINCT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
# run query safely
pollutants_with_zero_value = open_aq.query_to_pandas_safe(query3)

# view the 5 first rows of the returned dataframe
pollutants_with_zero_value.head()
# save dataframe as .csv
pollutants_with_zero_value.to_csv("pollutants_with_zero_value.csv")
