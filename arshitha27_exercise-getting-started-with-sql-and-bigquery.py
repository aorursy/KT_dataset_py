# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
from google.cloud import bigquery



# setting up the client - kinda like a middleman between jupyter notebook and bigquery 

# creating a "Client" object 

client = bigquery.Client()



# constructing a reference to the chicago crime dataset 

# client.dataset might seem like its returning the dataset but its actually returning a dataset reference

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# API request - fetching the actual dataset

dataset = client.get_dataset(dataset_ref)





# Write the code you need here to figure out the answer



# 

tables = list(client.list_tables(dataset))

num_tables = len(tables)  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

# q_1.solution()
# Write the code to figure out the answer

# constructing a reference to the 'crime' table

table_ref = dataset_ref.table('crime')



# getting the table 

# API request - fetch the table

crime = client.get_table(table_ref)



# schema_subset = [col for col in hn_full.schema if col.name in ('by', 'title', 'time')]



timestamp_fields = [col.name for col in crime.schema if col.field_type == 'TIMESTAMP']

print(timestamp_fields)

num_timestamp_fields = len(timestamp_fields) # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer

crime.schema
fields_for_plotting = ['x_coordinate', 'y_coordinate'] # Put your answers here



q_3.check()
q_3.hint()

q_3.solution()
# Scratch space for your code

client.list_rows(crime, max_results = 10).to_dataframe()