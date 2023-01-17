# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex2 import *



# import package with helper functions 

import bq_helper



# create a helper object for this dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")



print("Setup Complete")



# print list of tables in this dataset (there's only one!)

print('Tables list: {}'.format(open_aq.list_tables()))



# print look at top few rows

open_aq.head('global_air_quality')

# Your Code Goes Here

first_query = """SELECT country

FROM `bigquery-public-data.openaq.global_air_quality`

WHERE unit = 'ppm'"""



first_results = open_aq.query_to_pandas_safe(first_query)



# View top few rows of results

print(first_results.head())



q_1.check()
#q_1.solution()
from google.cloud import bigquery

client = bigquery.Client()



zero_pollution_query = """

                       SELECT *

                       FROM `bigquery-public-data.openaq.global_air_quality`

                       WHERE value = 0

                       """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(zero_pollution_query, job_config=safe_config)



zero_pollution_results = query_job.to_dataframe()



#zero_pollution_results = open_aq.query_to_pandas_safe(first_query)



#print(zero_pollution_results.head())



q_2.check()
#q_2.solution()