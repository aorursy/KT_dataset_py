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
[t.table_id for t in client.list_tables(dataset)]
num_tables = len([t.table_id for t in client.list_tables(dataset)])  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
table_ref = dataset.table('crime')

table = client.get_table(table_ref)

[field for field in table.schema if field.field_type == 'TIMESTAMP']
num_timestamp_fields = len([field for field in table.schema if field.field_type == 'TIMESTAMP']) # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
[field for field in table.schema if 'location' in field.description]
fields_for_plotting = ['latitude', 'longitude'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
client.list_rows(table, max_results=5).to_dataframe()