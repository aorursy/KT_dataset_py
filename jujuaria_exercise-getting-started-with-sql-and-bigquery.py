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



tables = list(client.list_tables(dataset))

num_tables = len(tables) # Store the answer as num_tables and then run this cell



q_1.check()
q_1.hint()

q_1.solution()
# Write the code to figure out the answer

for table in tables:  

    print(table.table_id)



table_ref = dataset_ref.table("crime")



crime = client.get_table(table_ref)



i=0

for s in crime.schema:

    i +=(s.field_type =="TIMESTAMP")

i
num_timestamp_fields = 2 # Put your answer here



q_2.check()
q_2.hint()

q_2.solution()
client.list_rows(crime, max_results=5).to_dataframe()
#fields_for_plotting = ["date", "location"] # Put your answers here

fields_for_plotting =['latitude', 'longitude']

q_3.check()
q_3.hint()

q_3.solution()
# Scratch space for your code