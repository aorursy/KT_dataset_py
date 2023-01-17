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
[print(table.table_id) for table in list(client.list_tables(dataset))]
num_tables = 1  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
table_crime_ref = dataset_ref.table("crime")

table_crime = client.get_table(table_crime_ref)



#[sum(column.dtype == 'TIMESTAMP') for column in table.columns]
num_timestamp_fields = sum([schema_filed.field_type == 'TIMESTAMP' for schema_filed in table_crime.schema])



q_2.check()
#q_2.hint()

#q_2.solution()
print([schema_filed.name  for schema_filed in table_crime.schema])
fields_for_plotting = ['latitude', 'longitude'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
client.list_rows(table_crime,max_results=5).to_dataframe()