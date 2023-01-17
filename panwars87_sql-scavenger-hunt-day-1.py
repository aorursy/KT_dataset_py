# import bigquery helper package
import bq_helper
# call BigQueryHelper method with active_project and dataset_name as parameter
# active_project = "bigquery-public-data"
# dataset_name = openaq
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
# method to list tables.
open_aq.list_tables()
# print few lines of the table (always a good practise)
open_aq.head("global_air_quality")
# build the query to pull all cities in US
query = """ select city from `bigquery-public-data.openaq.global_air_quality` 
where country='US' """
# execute the query and store in a variable
us_cities = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
# run sample data
us_cities.head()
# get total count by city
us_cities.city.value_counts().head()
non_ppm_country_query = """
select country, unit from `bigquery-public-data.openaq.global_air_quality`
where unit != 'ppm'
"""
non_ppm_country = open_aq.query_to_pandas_safe(non_ppm_country_query, max_gb_scanned=0.1)
non_ppm_country.head()
non_ppm_country.country.value_counts().head()
zero_pollutant_query = """
select pollutant, value from `bigquery-public-data.openaq.global_air_quality`
where value = 0
"""
# function to get the estimate size of queried data
open_aq.estimate_query_size(zero_pollutant_query)
zero_pollutant = open_aq.query_to_pandas_safe(zero_pollutant_query, max_gb_scanned=0.1)
zero_pollutant.head()
zero_pollutant.pollutant.value_counts().head()
