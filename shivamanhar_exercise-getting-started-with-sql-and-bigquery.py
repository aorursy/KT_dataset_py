# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")



#Other packages

import matplotlib.pyplot as plt

%matplotlib inline
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "chicago_crime" dataset

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")

'''

# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



#test programming

table_ref = dataset_ref.table('full')

table = client.get_table(table_ref)



table.schema



client.list_rows(table, max_results=5).to_dataframe()'''
print(len(list(client.list_tables(dataset_ref))))
# Write the code you need here to figure out the answer

for table_name in  list(client.list_tables(dataset_ref)):

    print(table_name.table_id)
table_ref = dataset_ref.table('crime')

table = client.get_table(table_ref)
table.schema
tables = list(client.list_tables(dataset_ref))
ds = client.list_rows(table, max_results=5).to_dataframe()

ds.head(5)
num_tables = len(list(client.list_tables(dataset_ref)))  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
ds.dtypes
# Write the code to figure out the answer
num_timestamp_fields = 2 # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer
fields_for_plotting = ['latitude', 'longitude'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code

ds.columns
plt.plot(ds['latitude'])



plt.plot(ds['longitude'])