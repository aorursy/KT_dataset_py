# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *





# create a helper object for our bigquery dataset

import bq_helper

chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                         dataset_name = "chicago_crime")

print("Setup Complete")
____ # Write the code you need here to figure out the answer
chicago_crime.list_tables()
num_tables = 1  # store the answer as num_tables and then run this cell



q_1.check()
# q_1.hint()

# q_1.solution()
____ # Write the code to figure out the answer

from google.cloud import bigquery

client = bigquery.Client()

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)

table_ref = dataset_ref.table("crime")

# API request - fetch the table

table = client.get_table(table_ref)

table.schema
num_timestamp_fields = 2 # put your answer here



q_2.check()
# q_2.hint()

# q_2.solution()
____ # Write the code here to explore the data so you can find the answer

client.list_rows(table, max_results=5).to_dataframe()
fields_for_plotting = ['x_coordinate', 'y_coordinate']



q_3.check()
# q_3.hint()

# q_3.solution()
# Scratch space for your code

# Preview the first five entries in the "by" column of the "full" table

client.list_rows(table, selected_fields=table.schema[19:22], max_results=5).to_dataframe()