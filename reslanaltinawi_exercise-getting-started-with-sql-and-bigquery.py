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

print(len(tables))
num_tables = 1  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer

crime_table_ref = dataset_ref.table('crime')



crime_table = client.get_table(crime_table_ref)



print(crime_table.schema)
timestamp_columns = [column for column in crime_table.schema if column.field_type == 'TIMESTAMP']

print(timestamp_columns)

print(len(timestamp_columns))
num_timestamp_fields = 2 # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
for column in crime_table.schema:

    print(column)

    print('=' * 20)
# Write the code here to explore the data so you can find the answer
fields_for_plotting = ['x_coordinate', 'y_coordinate'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code