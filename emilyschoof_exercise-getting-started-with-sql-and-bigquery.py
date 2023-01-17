# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
from google.cloud import bigquery



# Create a "Client" object - blueprint of what a client should be

client = bigquery.Client()



# Construct a reference to the "chicago_crime" dataset - create a relationship between client and 'chicago-crime'

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))

print(len(tables))



# Print names of all tables in the dataset 

for table in tables:  

    print(table.table_id)
num_tables = len(tables)  # Store the answer as num_tables and then run this cell



q_1.check() # Check if calculation and recall of number tables in dataset is correct
q_1.hint() # Get hint from query to solve a solution

#q_1.solution() # Get entire solution to solve for a specific query question
# Construct a reference to the "crime" table

table_ref = dataset_ref.table("crime")



# API request - fetch the table

table = client.get_table(table_ref)



# Print information on all the columns in the "crime" table in the "chicago_crime" dataset

print(table.schema)
num_timestamp_fields = 2 # Put your answer here



q_2.check()
q_2.hint()

#q_2.solution()
# Preview the first five lines of the "crime" table

client.list_rows(table, max_results=5).to_dataframe()
fields_for_plotting = ['latitude', 'longitude'] # Latitude/Longitude offer the geographic location of crimes



q_3.check()
q_3.hint()

#q_3.solution()
# Scratch space for your code