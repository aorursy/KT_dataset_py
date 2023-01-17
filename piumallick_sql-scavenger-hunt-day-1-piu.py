# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# Your code goes here :)
# Answer to Question 1 : Which countries use a unit other than ppm to measure any type of pollution?

query_1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """

pollutants_not_ppm = open_aq.query_to_pandas_safe(query_1)

pollutants_not_ppm.country.value_counts().head(10)
# Your code goes here :)
# Answer to Question 2 : Which pollutants have a value of exactly 0?

query_2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
pollutants_having_value_0 = open_aq.query_to_pandas_safe(query_2)

pollutants_having_value_0.pollutant.value_counts().head(10)
