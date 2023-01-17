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
# list all the tables in the "chicago_crime" dataset. 
tables= list(client.list_tables(dataset))
# counting how many tables in the dataset
len(tables)

num_tables =1 
q_1.check()
# Write the code to figure out the answer
# Print the names in the dataset
for table in tables:
    print(table.table_id)
 # construct a reference to the "crime" table
table_ref = dataset_ref.table("crime")
# fetch the table
table = client.get_table(table_ref)

#print information on all the columns in the "crime" table
table.schema
num_timestamp_fields = 2 # Put your answer here
# Check your answer
q_2.check()
#q_2.hint()
#q_2.solution()
# Write the code here to explore the data so you can find the answer
# printing the first five lines
client.list_rows(table, max_results=5).to_dataframe()
fields_for_plotting = ['latitude', 'longitude'] # Put your answers here

# Check your answer
q_3.check()
#q_3.hint()
#q_3.solution()
# Scratch space for your code