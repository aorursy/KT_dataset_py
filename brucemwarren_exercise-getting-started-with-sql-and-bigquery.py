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
tables = list(client.list_tables(dataset)) # Create iterable list

count = 0 # Set counter

for table in tables:

    count += 1 # Count tables

    print(table.table_id) # Print tables as an editcheck

print(count) # As in solution, this should have been more Pythonic - print(len(tables))
num_tables = 1  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

q_1.solution()
# Fetch the crime dataset

table_ref = dataset_ref.table("crime")

table = client.get_table(table_ref)



#Print information on table

print(table.schema)



# Print first five lines and count

client.list_rows(table, max_results=5).to_dataframe()

num_timestamp_fields = 2 # Put your answer here



q_2.check()
#q_2.hint()

q_2.solution()
# Write the code here to explore the data so you can find the answer

# Need to look at the schema again for location data

print(table.schema)
fields_for_plotting = ['latitude', 'longitude'] # Put your answers here



q_3.check()
#q_3.hint()

q_3.solution()
# Scratch space for your code

client.list_rows(table, max_results=5).to_dataframe()

# The location is a tuple of the lat and long