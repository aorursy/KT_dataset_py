# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head('global_air_quality')
query= '''SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            '''
countries_not_ppm=open_aq.query_to_pandas_safe(query)
print(countries_not_ppm)
# Your Code Goes Here
query_2='''SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0.00
            '''

pollutants_0=open_aq.query_to_pandas_safe(query_2)
print(pollutants_0)