# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
open_aq.table_schema("global_air_quality")
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
# Your coe goes here :)
query2 = '''
select distinct country from `bigquery-public-data.openaq.global_air_quality`
where unit!='ppm'
'''
no_ppm = open_aq.query_to_pandas_safe(query2)

query3 = '''
select pollutant,sum(value) as tol
from `bigquery-public-data.openaq.global_air_quality`
where country in ("LT")
group by pollutant
having tol = 0

'''
zero_plt = open_aq.query_to_pandas_safe(query3)
