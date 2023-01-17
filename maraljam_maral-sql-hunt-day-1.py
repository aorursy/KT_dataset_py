# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

open_aq.head("global_air_quality")

query = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
non_ppm_cons = open_aq.query_to_pandas_safe(query)
non_ppm_array = non_ppm_cons.country.unique()
print(non_ppm_array, non_ppm_array.size)

#Question2

query_2 = """select pollutant, value
from `bigquery-public-data.openaq.global_air_quality`
where value = 0
"""	

zeros = open_aq.query_to_pandas_safe(query_2)
zeros_array = zeros.pollutant.unique()
print (zeros_array)