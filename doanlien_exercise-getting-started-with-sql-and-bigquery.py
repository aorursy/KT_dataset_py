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

#list all tables in the "chicago_crime" dataset

# List all the tables in the "chicago_crime" dataset

tables = list(client.list_tables(dataset))



# Print number of tables in the dataset

print(len(tables))



  # Store the answer as num_tables and then run this cell



q_1.check()
q_1.hint()

q_1.solution()
# Write the code to figure out the answer

table_ref = dataset_ref.table("crime")

table = client.get_table(table_ref)

print (table.schema)

num_timestamp_fields = 0

for t in table.schema:

    if(t.field_type == 'TIMESTAMP'):

        num_timestamp_fields = num_timestamp_fields + 1

print (num_timestamp_fields)

num_timestamp_fields = ____ # Put your answer here



q_2.check()
q_2.hint()

q_2.solution()
# Write the code here to explore the data so you can find the answer

df = client.list_rows(crime_table, max_results=25).to_dataframe()

for column in df.columns:

    print(column)





fields_for_plotting = ['latitude', 'longitude']
fields_for_plotting = [____, ____] # Put your answers here



q_3.check()
q_3.hint()

q_3.solution()
# Scratch space for your code