# set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup is completed")
from google.cloud import bigquery



# create a "Client" object

client = bigquery.Client()



# construct a reference to the "chicago_crime" dataset

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# fetch the dataset - API request

dataset = client.get_dataset(dataset_ref)
# write the code you need here to figure out the answer

tables = list(client.list_tables(dataset))

[table.table_id for table in tables]
# store the answer as num_tables and then run this cell

num_tables = len(tables)



q_1.check()
# for a hint or the solution, uncomment the appropriate line below.

# q_1.hint()

# q_1.solution()
# construct a reference to the "crime" table

table_ref = dataset_ref.table("crime")



# fetch the table

table = client.get_table(table_ref)
# print information on all the columns in the "crime" table

table.schema
num_timestamp_fields = 2



q_2.check()
# for a hint or the solution, uncomment the appropriate line below.

# q_2.hint()

# q_2.solution()
# Write the code here to explore the data so you can find the answer
fields_for_plotting = ['latitude', 'longitude']



q_3.check()
# For a hint or the solution, uncomment the appropriate line below.

# q_3.hint()

# q_3.solution()
# Scratch space for your code

client.list_rows(table, max_results=5).to_dataframe()