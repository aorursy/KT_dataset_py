from google.cloud import bigquery
# Create a "Client" object

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "full" table

table_ref = dataset_ref.table("full")



# API request - fetch the table

table = client.get_table(table_ref)
# Print information on all the columns in the "full" table in the "hacker_news" dataset

table.schema
# Preview the first five lines of the "full" table

bq_file = client.list_rows(table, max_results=25).to_dataframe()



# save file

import pandas as pd

with open("my_file.csv", 'w') as file:

    file.write(bq_file.to_csv())
# Preview the first five entries in the "by" column of the "full" table

client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()