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
num_tables = len(list(client.list_tables(dataset)))# Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the

table_ref = dataset_ref.table("crime")



# API request - fetch the table

table = client.get_table(table_ref)



# Print information on all the columns in the "crime" table in the "chicago_crime" dataset

print(table.schema) 
num_timestamp_fields = 2# Put your answer here



q_2.check()
# Write the code here to explore the data so you can find the answer
fields_for_plotting = ['longitude', 'latitude'] # Put your answers here



q_3.check()
# Scratch space for your code