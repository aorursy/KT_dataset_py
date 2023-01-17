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
query = """SELECT country  
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            GROUP BY country
        """

q1 = open_aq.query_to_pandas_safe(query)
q1.head()
q1.shape
query = """SELECT pollutant  
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            GROUP BY pollutant
        """

q2 = open_aq.query_to_pandas_safe(query)
q2
q2.shape
query = """
    SELECT country, count(distinct(unit)) as count
    FROM `bigquery-public-data.openaq.global_air_quality`    
    GROUP BY country
    HAVING count > 1
"""
q3 = open_aq.query_to_pandas_safe(query)
q3
query = """
SELECT DISTINCT country,unit FROM `bigquery-public-data.openaq.global_air_quality` WHERE country in (SELECT country FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != 'ppm') order by country"""
q4 = open_aq.query_to_pandas_safe(query)
q4
query1 = """
    SELECT country, unit 
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit = 'ppm'
 """
query2 = """
    SELECT country, unit 
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit != 'ppm'
"""
res_q =  """
    SELECT country, unit 
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit = 'ppm'
    INTERSECT ALL
    SELECT country, unit 
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit != 'ppm'
"""
q4 = open_aq.query_to_pandas_safe(res_q)
q4