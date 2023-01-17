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

first_query = """SELECT DISTINCT country

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE unit = 'ppm'

        """

first_results = open_aq.query_to_pandas_safe(first_query)



# View top few rows of results

print(first_results.head())



q_1.check()
#q_1.solution()
# Your Code Goes Here



zero_pollution_query = """SELECT *

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE value = 0

        """

zero_pollution_results = open_aq.query_to_pandas_safe(zero_pollution_query)



print(zero_pollution_results.head())



q_2.check()
#q_2.solution()