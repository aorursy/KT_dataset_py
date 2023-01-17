from google.cloud import bigquery

import numpy as np

import pandas as pd



# Create a "Client" object

client =  bigquery.Client()



dataset_ref = client.dataset("hacker_news", project="bigquery-public-data") 

dataset= client.get_dataset(dataset_ref) # dataset is a collection of tables and a project is a collectio of datasets
# see contents of the table

# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)
# create a reference to a specific table e.g full table



table_ref = dataset_ref.table("full")



# API request - fetch the table

table = client.get_table(table_ref) # bcos we r getting a table and not a dataset in this case



table.schema





import google

#dir(google.cloud.bigquery)
# Preview the first five lines of the "full" table

fullData = client.list_rows(table, max_results=25).to_dataframe() # it is converted to a dataframe



#saving this to a .csv file

with open("fullData.csv", "w") as file:

    file.write(fullData.to_csv())


