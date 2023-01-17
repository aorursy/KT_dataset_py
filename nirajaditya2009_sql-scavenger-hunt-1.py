# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
not_ppm_query = """SELECT country 
                   FROM `bigquery-public-data.openaq.global_air_quality`
                   WHERE unit != "ppm" """
open_aq.estimate_query_size(not_ppm_query)
not_ppm = open_aq.query_to_pandas_safe(not_ppm_query)      
print ("Few Countries which doesnt use the unit PPM")
not_ppm.head()
print("Top Countries")
not_ppm.country.value_counts().head()
polls_query = """SELECT pollutant
                   FROM `bigquery-public-data.openaq.global_air_quality` 
                   WHERE value = 0"""
print ("Estimated Query Size : {} GB".format(open_aq.estimate_query_size(polls_query)))
zero_polls = open_aq.query_to_pandas_safe(polls_query)  
zero_polls.shape
zero_polls.pollutant.value_counts()