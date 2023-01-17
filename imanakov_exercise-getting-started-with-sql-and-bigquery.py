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
len(list(client.list_tables(dataset)))
num_tables = 1  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
crime_ref = dataset_ref.table('crime')

crime = client.get_table(crime_ref)

w_ts = [col for col in crime.schema if col.field_type == 'TIMESTAMP']

display(crime.schema)

len(w_ts)
num_timestamp_fields = 2 # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer
fields_for_plotting = ['x_coordinate', 'y_coordinate'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
client.list_rows(crime, max_results=5).to_dataframe()