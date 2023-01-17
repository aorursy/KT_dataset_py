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

tables=list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)

print(len(tables))

table_ref = dataset_ref.table("crime")  # Store the answer as num_tables and then run this cell

table=client.get_table(table_ref)



num_tables=1

q_1.check()
#q_1.hint()

#q_1.solution()

# Write the code to figure out the answer

print(table.schema)

#schema tells about specific field or colunm

print(client.list_rows(table, max_results=5).to_dataframe())

client.list_rows(table, selected_fields=table.schema[:7], max_results=5).to_dataframe()
num_timestamp_fields = 2 # from schema



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer

print(len(table.schema))

client.list_rows(table, selected_fields=table.schema[:22], max_results=5).to_dataframe()

fields_for_plotting = ['latitude', 'longitude'] # Put your answers here



q_3.check()
q_3.hint()

#q_3.solution()
# Scratch space for your code