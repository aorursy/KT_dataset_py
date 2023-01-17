# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex2 import *

from google.cloud import bigquery







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



query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        """

# Create a "Client" object

client = bigquery.Client()



# Set up the query

query_job = client.query(query)



# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()



us_cities.head()
# Your Code Goes Here

# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex2 import *

from google.cloud import bigquery







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



first_query = """

        SELECT DISTINCT country

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE unit = 'ppm'

        """

# Create a "Client" object

client = bigquery.Client()



# Set up the query

query_job = client.query(first_query)



# API request - run the query, and return a pandas DataFrame

first_results = query_job.to_dataframe()









# View top few rows of results

print(first_results.head())



q_1.check()
#q_1.solution()
# Your Code Goes Here



zero_pollution_query = """SELECT *

                        FROM `bigquery-public-data.openaq.global_air_quality`

                        WHERE value = 0"""

# Create a "Client" object

client = bigquery.Client()



# Set up the query

query_job = client.query(zero_pollution_query)



zero_pollution_results = query_job.to_dataframe()



print(zero_pollution_results.head())



q_2.check()
#q_2.solution()