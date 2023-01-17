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
tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)

print(len(tables))    
num_tables = 1

q_1.check()
q_1.hint()

q_1.solution()


# Construct a reference to the "crime" table

table_ref = dataset_ref.table("crime")

#fetch the table

table = client.get_table(table_ref)

print(table.schema)



num_timestamp_fields = 2 

q_2.check()
num_timestamp_fields = 2 

q_2.check()
q_2.hint()

q_2.solution()
print (table.schema)

client.list_rows(table,max_results=10).to_dataframe()
fields_for_plotting = ['latitude','longitude'] 

q_3.check()
q_3.hint()

q_3.solution()
# Scratch space for your code

client.list_rows(table,selected_fields= table.schema[:1], max_results=10).to_dataframe()