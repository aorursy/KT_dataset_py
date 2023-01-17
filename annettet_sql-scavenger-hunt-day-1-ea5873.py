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
query1 = """SELECT DISTINCT UNIT
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
unit_dist = open_aq.query_to_pandas_safe(query1)
unit_dist

# Your code goes here :)
query2 = """SELECT DISTINCT COUNTRY
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit!='ppm'
            ORDER BY COUNTRY
        """
cntr = open_aq.query_to_pandas_safe(query2)
cntr

query3 = """SELECT COUNTRY,
SUM(CASE WHEN unit='ppm' then 1 else 0 End) as ppm_cnt,
SUM(CASE WHEN unit='µg/m³' then 1 else 0 End) as ugm3_cnt
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY COUNTRY
            ORDER BY COUNTRY
        """
unit_cnt = open_aq.query_to_pandas_safe(query3)
unit_cnt
# Following countries only measure polluntant in µg/m³:)
query4 = """SELECT COUNTRY,
SUM(CASE WHEN unit='ppm' then 1 else 0 End) as ppm_cnt,
SUM(CASE WHEN unit='µg/m³' then 1 else 0 End) as ugm3_cnt
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY COUNTRY
            HAVING ppm_cnt=0 and ugm3_cnt > 0
            ORDER BY COUNTRY
        """
ugm3_only = open_aq.query_to_pandas_safe(query4)
ugm3_only
