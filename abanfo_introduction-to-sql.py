# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "chicago_crime" dataset

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)

# Write the code you need here to figure out the answer

tables = list(client.list_tables(dataset))

num_tables = [table.table_id for table in tables]  # Store the answer as num_tables and then run this cell

num_tables =1

q_1.check()
# Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("crime")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()


# Write the code to figure out the answer



table.schema
# the schema of the table, or the meta data

num_timestamp_fields = 2 # Put your answer here



q_2.check()
fields_for_plotting = ['latitude','longitude'] # Put your answers here



q_3.check()