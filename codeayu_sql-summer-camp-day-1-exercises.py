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

print(f'no. of tables : {len(tables)}')

print(tables[0].table_id)
num_tables = 1  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer

table_ref = dataset_ref.table("crime")

table_data = client.get_table(table_ref)

print(*(i for i in table_data.schema if 'TIMESTAMP'==i.field_type),sep='\n')
num_timestamp_fields = 2# Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer

table_data.schema
fields_for_plotting = ['x_coordinate', 'y_coordinate'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code