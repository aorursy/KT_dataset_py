# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
# Your Code Goes Here
#select distinct will only select one row 
query = """SELECT DISTINCT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != "ppm" """
output = open_aq.query_to_pandas_safe(query)
output.country
# Your Code Goes Here
query = """SELECT pollutant,location
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0 """
pollutant_value = open_aq.query_to_pandas_safe(query)
pollutant_value.tail(10)
#this question is confusing. output one column with only pollutants seems confusing so i included location as well.