# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
from time import time
# question 1
query = """
    select country
    from `bigquery-public-data.openaq.global_air_quality`
    where unit != 'ppm'
    """
start = time()
countries = open_aq.query_to_pandas_safe(query)
countries_list = list(countries['country'].unique())
end = time()
print('Execution time: ', round(end - start, 5))
print(countries_list)
from time import time
# question 1
query = """
    select country
    from `bigquery-public-data.openaq.global_air_quality`
    where unit != 'ppm'
    group by country
    """
start = time()
countries = open_aq.query_to_pandas_safe(query)
countries_list = list(countries['country'])
end = time()
print('Execution time: ', round(end - start, 5))
print(countries_list)
# question 2
query = """
    select pollutant
    from `bigquery-public-data.openaq.global_air_quality`
    where value = 0.0
    group by pollutant
    """
pollutants = open_aq.query_to_pandas_safe(query)
print(list(pollutants['pollutant']))