from google.cloud import bigquery

import pandas as pd
client = bigquery.Client()

# Construct a reference to the dataset

dataset_ref = client.dataset("cms_medicare", project="bigquery-public-data")

# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "cms_medicare" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset

for table in tables:

    print(table.table_id)
hospital_general_info_ref = dataset_ref.table("hospital_general_info")

hospital_general_table = client.get_table(hospital_general_info_ref)
hospital_general_table.schema
# Preview the first five lines of the "full" table

client.list_rows(table, max_results=5).to_dataframe()