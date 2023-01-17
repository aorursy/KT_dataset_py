# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
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
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
num_tables = 1  



q_1.check()
# Construct a reference to the "crime" table

table_ref = dataset_ref.table("crime")



# API request - fetch the table

table = client.get_table(table_ref)
# Print information on all the columns in the "crime" table in the "chicago_crime" dataset

table.schema
num_timestamp_fields = 2



q_2.check()
fields_for_plotting =['latitude', 'longitude']



q_3.check()
client.list_columns(table, max_results=22).to_dataframe()


q_3.solution()
list_rows()
client.list_rows(table, selected_fields=table.schema[20], max_results=5).to_dataframe()