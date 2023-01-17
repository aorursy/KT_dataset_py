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
# List all the tables in the "chicago_crime" dataset

tables = list(client.list_tables(dataset))



#Sol1

num_tables=0

for table in tables:  

    num_tables+=1

    print(table.table_id)





# Sol 2

num_tables=len(tables)

print("num_tables = " + str(num_tables))



q_1.check()
# Construct a reference to the "crime" table

table_ref = dataset_ref.table("crime")



# API request - fetch the table

table = client.get_table(table_ref)



# Print information on all the columns in the "crime" table in the "chicago_crime" dataset

print("num_timestamp_fields = ")

print(table.schema)



table.schema.types
num_timestamp_fields = 2



q_2.check()
client.list_rows(table, max_results=5).to_dataframe()
client.list_rows(table, selected_fields=table.schema[15:17], max_results=5).to_dataframe()

fields_for_plotting = ["x_coordinate", "y_coordinate"] # Put your answers here



q_3.check()
# Scratch space for your code