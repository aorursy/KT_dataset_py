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
num_tables = len(list(client.list_tables(dataset)))  # Store the answer as num_tables and then run this cell

print(num_tables)



q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer

# List all the tables in the "chicago_crime" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset

for table in tables:  

    print(table.table_id)



# Construct a reference to the "crime" table

table_ref = dataset_ref.table("crime")

# API request - fetch the table

table = client.get_table(table_ref)

# Print information on all the columns in the "crime" table in the "chicago_crime" dataset

table.schema
num_timestamp_fields = 2 # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer
fields_for_plotting = ["latitude", "longitude"] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code



# Preview the first five lines of the "crime" table

client.list_rows(table, max_results=5).to_dataframe()