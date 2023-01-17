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

count_tables = 0

tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)

    count_tables += 1
num_tables = count_tables  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer

table_ref = dataset_ref.table("crime")

table = client.get_table(table_ref)

frame_ex = client.list_rows(table, selected_fields=table.schema[:], max_results=5).to_dataframe()

frame_ex.dtypes
import pandas as pd

count = 0

for x in frame_ex.dtypes:

    if isinstance(x, pd.core.dtypes.dtypes.DatetimeTZDtype):

        count += 1

print(count)
num_timestamp_fields = count # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer

frame_ex
fields_for_plotting = ["latitude", "longitude"] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code

client.list_rows(table, selected_fields=table.schema[:], max_results=5).to_dataframe()