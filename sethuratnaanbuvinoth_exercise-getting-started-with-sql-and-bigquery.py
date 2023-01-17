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
# List all the tables in the "Chicago Crime" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset 

for table in tables:

    print(table.table_id)
num_tables = len(tables)  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
# Construct a reference to the "crime" table

table_ref = dataset_ref.table("crime")



# API request - fetch the table

crime_table = client.get_table(table_ref)



# Print information on all the columns in the "crime" table in the "Chicago Crime" dataset

crime_table.schema
num_timestamp_fields = 0



for t in crime_table.schema:

    if t.field_type == 'TIMESTAMP':

        num_timestamp_fields +=1



print(num_timestamp_fields) # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Display first five rows of the table to get an idea about the columns

client.list_rows(crime_table, max_results = 5).to_dataframe()
# Latitude and Longitude gives the location of each crime. That is the last two fields of the table.

table_name = client.list_rows(crime_table, selected_fields = crime_table.schema[-3:-1], max_results = 5).to_dataframe()



# table_name has last two fields

fields_for_plotting = list(table_name)

print (fields_for_plotting)  



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code