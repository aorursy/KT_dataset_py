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

tables=list(client.list_tables(dataset))

actual_list=[]

for table in tables:

    actual_list.append(table.table_id)

    print(table.table_id)
num_tables = len(actual_list)  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer

crime_ref=dataset.table('crime')

crime_table=client.get_table(crime_ref)

n_timestamps=0

for schema in crime_table.schema:

    print(schema)

    if 'TIMESTAMP' in str(schema):

        n_timestamps=n_timestamps+1

print(n_timestamps)
num_timestamp_fields = n_timestamps # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer

import pandas as pd

df=client.list_rows(crime_table,max_results=25).to_dataframe()

df.head()
fields_for_plotting = ['x_coordinate', 'y_coordinate'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
#saving the data to a csv for later

with open('chicago_crime_subsample.csv','w') as file:

    file.write(df.to_csv())