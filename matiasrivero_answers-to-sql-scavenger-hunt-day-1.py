# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# let's go with the first question:
# query to select all the countries from the "country" column where the
# "unit" column is NOT "ppm"
# use the DISTINCT keyword to get only distinct (not repeated) values (in this case: countries)
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
not_ppm = open_aq.query_to_pandas_safe(query)
# we now simply list all the countries which fulfill the previous requirement
# this list gives us insigt about all those countries which uses another measurement unit than ppm
not_ppm
# let's go now with the second question:
# query to select all the pollutants from the "pollutant" column where the
# "value" column is "0"
# we use again the DISTINCT keyword so we only get those pollutants which are in 0 ppm
query2 = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollutant_in_zero = open_aq.query_to_pandas_safe(query2)
# we now list all the pollutants which have a 0 value
# keep in mind that, as we have used the DISTINCT keyword, 
# this list represents all those pollutants that are 0 at least in just a single location
# but it happens that the pollutant so2 is 0 in 
# Ate, Lima, Peru, but 0.044 in Huachipa, Lima, Peru
pollutant_in_zero
# this allows to check that each pollutant appears only once
pollutant_in_zero.pollutant.value_counts()
# it can be more interesting to count how many times a pollutant is equal to 0
# this is possible removing the DISTINCT keyword in the query
query3 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollutant_in_zero_all = open_aq.query_to_pandas_safe(query3)
# let's count the number each pollutant is zero
pollutant_in_zero_all.pollutant.value_counts()
