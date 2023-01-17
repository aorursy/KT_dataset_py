# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
# To use BigQuery, we'll import the Python package below:

from google.cloud import bigquery



# Create a "Client" object which will play a central role in retrieving information from BigQuery datasets.

client = bigquery.Client()



# In BigQuery, each dataset is contained in a corresponding project.

# Our chicago_crime dataset is contained in the bigquery-public-data project. To access the dataset:

# 1. Construct a reference to the "chicago_crime" dataset with the dataset() method

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# We use the get_dataset() method, along with the reference we just constructed, to fetch the dataset.

# 2. API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Write the code you need here to figure out the answer

# Every dataset is just a collection of tables. We use the list_tables() method to list the tables in the dataset.



# List all the tables in the "chicago_crime" dataset

tables=list(client.list_tables(dataset))



# Printing number of tables in the dataset

print(len(tables))



# Listing the names of all the tables in the dataset

for table in tables:

    print(table.table_id)
num_tables = 1  # Store the answer as num_tables and then run this cell



q_1.check()
q_1.hint()

q_1.solution()
# Write the code to figure out the answer

# Similar to how we fetched a dataset, we can fetch a table. We fetch the crime table in the chicago_crime dataset.



# 1. Construct a reference to the "crime" table with the table() method.

table_ref = dataset_ref.table("crime")



# 2. We use the get_table() method, along with the reference we just constructed, to fetch the table.

# API request - fetch the table

table = client.get_table(table_ref)



# The structure of a table is called its schema.

# Print information on all the columns in the "crime" table in the "chicago_crime" dataset

table.schema
num_timestamp_fields = 2 # Put your answer here



q_2.check()
q_2.hint()

q_2.solution()
# Write the code here to explore the data so you can find the answer

# list_rows() method to check just the first five lines of of the crime table.

# This returns a BigQuery RowIterator object that can be converted to a pandas DataFrame with the to_dataframe() method.

client.list_rows(table, max_results=5).to_dataframe()
# The list_rows() method will also let us look at just the information in a specific column.

client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
fields_for_plotting = ['latitude','longitude'] # Put your answers here



q_3.check()
q_3.hint()

q_3.solution()
# Scratch space for your code