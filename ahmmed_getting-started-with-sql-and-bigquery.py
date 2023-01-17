import pandas as pd

from google.cloud import bigquery
client = bigquery.Client()
# Create a dataset ref 

dataset_ref = client.dataset("chicago_crime", project= "bigquery-public-data")



# fetch the dateset

dataset = client.get_dataset(dataset_ref)
# List all the table

tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)
# Create ref for "crime" table

crime_ref = dataset_ref.table("crime")

# Fetch table data

crime = client.get_table(crime_ref)

crime.schema
df = client.list_rows(crime, max_results=20).to_dataframe()

df.to_csv("crime.csv", index=False)
import os

os.listdir()