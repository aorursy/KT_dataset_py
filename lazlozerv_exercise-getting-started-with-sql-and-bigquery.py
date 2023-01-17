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
#Number of tables in the "chicago_crime" dataset

num_tables = len(list(client.list_tables(dataset)))  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
print(table.schema)
# Construct a reference to the "full" table

table_ref = dataset_ref.table("crime")



# API request - fetch the table

table = client.get_table(table_ref)



# Number of columns with 'TIMESTAMP' data

num_timestamp_fields = 0

for col in table.schema:

    if col.field_type == 'TIMESTAMP':

        num_timestamp_fields += 1 



q_2.check()
q_2.hint()

q_2.solution()
print(table.schema)
# Standard answer

fields_for_plotting = ['latitude', 'longitude']

# Another correct answer

# fields_for_plotting = ['x_coordinate', 'y_coordinate']



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code

client.list_rows(table, max_results=5).to_dataframe()

# Location is a tuple,pair of (latitude, longitude)