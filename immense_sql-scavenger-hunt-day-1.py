# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
query = "select DISTINCT country,unit  from `bigquery-public-data.openaq.global_air_quality` where unit != 'ppm' "

countries_No_ppm = open_aq.query_to_pandas_safe(query)

countries_No_ppm.head()




#query = "SELECT pollutant from `bigquery-public-data.openaq.global_air_quality` where value= 0"
#pollutant_value0 = open_aq.query_to_pandas_safe(query)

#pollutant_value0.pollutant.value_counts().head()

query = "SELECT DISTINCT pollutant from `bigquery-public-data.openaq.global_air_quality` where value= 0"
pollutant_value0 = open_aq.query_to_pandas_safe(query)

pollutant_value0.pollutant


