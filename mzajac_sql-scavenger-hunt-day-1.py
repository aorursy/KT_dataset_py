# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# First Task
query_country="""SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' """
coutries=open_aq.query_to_pandas_safe(query_country)
print('Which countries use a unit other than ppm to measure any type of pollution?')
print (coutries)
# Second Task
query_poll="""SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
            """
pollutant=open_aq.query_to_pandas_safe(query_poll)
print('Which pollutants have a value of exactly 0?')
print (pollutant)