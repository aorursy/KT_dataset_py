# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

# query to select countries  - use a unit other than ppm to measure any type of pollution
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
query1_country = open_aq.query_to_pandas_safe(query1)


#  query to select pollutants - have a value of exactly 0
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
query2_pollutant = open_aq.query_to_pandas_safe(query2)

# resulting output of query1
query1_country.country.head()
# resulting output of query2
query2_pollutant.pollutant.head()
