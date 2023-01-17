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

table_list = list(client.list_tables(dataset))

print(len(table_list))
num_tables = len(list(client.list_tables(dataset)))  # Store the answer as num_tables and then run this cell



# Check your answer

q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer

# for t in table_list:

#     print(t.table_id)



# create a reference to table we need

table_crime_ref = dataset_ref.table('crime')

# now fetch that table

table_crime = client.get_table(table_crime_ref)



# let us observe the schema of this table

table_crime.schema
num_timestamp_fields = 2 # Put your answer here



# Check your answer

q_2.check()
# q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer

client.list_rows(table_crime, max_results=25).to_dataframe()
fields_for_plotting = ['latitude','longitude'] # Put your answers here



# Check your answer

q_3.check()
#q_3.hint()

# q_3.solution()
# Scratch space for your code