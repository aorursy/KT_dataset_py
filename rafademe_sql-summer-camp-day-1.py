# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
import pandas as pd
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "chicago_crime" dataset

chicago_crime_dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# API request - fetch the dataset

chicago_crime_dataset = client.get_dataset(chicago_crime_dataset_ref)
# Write the code you need here to figure out the answer

chicago_crime_tables = list(client.list_tables(chicago_crime_dataset))



for table in chicago_crime_tables:  

    print(table.table_id)



# Print number of tables in the dataset

print(len(chicago_crime_tables))



num_tables = len(chicago_crime_tables)  # Store the answer as num_tables and then run this cell



q_1.check()
# q_1.hint()

# q_1.solution()
# Write the code to figure out the answer



# Construct a reference to the "crime" table

crime_table_ref = chicago_crime_dataset_ref.table("crime")



# API request - fetch the table

crime_table = client.get_table(crime_table_ref)



# Print information on all the columns in the "crime" table in the "chicago_crime" dataset

crime_table.schema

num_timestamp_fields = 2 # Put your answer here



q_2.check()
# q_2.hint()

# q_2.solution()
# Write the code here to explore the data so you can find the answer



# Print information on all the columns in the "crime" table in the "chicago_crime" dataset

crime_table.schema
fields_for_plotting = ["latitude", "longitude"] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code



# Preview the first five lines of the "full" table as a pandas DataFrame

client.list_rows(crime_table, max_results=5).to_dataframe()