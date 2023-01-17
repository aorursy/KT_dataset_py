# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

#Testing if all pm25 pollutants have 0.00 value
query = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant = "pm25" and value = 0.00
        """
rows1 = open_aq.query_to_pandas_safe(query)
rows1.head()
# Rows where pm25 not equal to zero
query6 = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant = "pm25" and value != 0.00
        """
rows2 = open_aq.query_to_pandas_safe(query6)
rows2.head()
# query to select all the items from the "city" column where the
# "country" column is "us"
query4 = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas_safe(query4)
# What five cities have the most measurements taken there?
#value_counts()
us_cities.city.value_counts().head()
#SCAVENGER HUNT
#Questions
# 1. Which countries use a unit other than ppm to measure any type of pollution?
#(Hint: to get rows where the value isn't something, use "!=")
# 2.Which pollutants have a value of exactly 0?
query2 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
countries_unit_not_eq_ppm = open_aq.query_to_pandas_safe(query2)
countries_unit_not_eq_ppm.head()
countries_unit_not_eq_ppm.country.value_counts().head()
#List of unique values (here countries)
countries_unit_not_eq_ppm.country.unique()
#Qs 2:  Which pollutants have a value of exactly 0
query3 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value != 0.00
        """
pollutant_value_zero = open_aq.query_to_pandas_safe(query3)
#First 5 rows of the datframe
pollutant_value_zero.pollutant.head()
#Unique values in the dataframe
pollutant_value_zero.pollutant.unique()