#import bigquery

from google.cloud import bigquery
client = bigquery.Client()
dataset_ref=client.dataset("hacker_news",project="bigquery-public-data")

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
table.schema
# Preview the first 30 lines of the "full" table

client.list_rows(table, max_results=30).to_dataframe()
# Preview the first 10 entries in the "by" column of the "full" table

client.list_rows(table, selected_fields=table.schema[:2], max_results=10).to_dataframe()