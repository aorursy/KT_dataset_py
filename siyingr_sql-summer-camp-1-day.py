from google.cloud import bigquery
# Create a "Client" object

client = bigquery.Client()
# Construct a reference to the "chicago_crime" dataset

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "chicago_crime" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there is one!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "crime" table

table_ref = dataset_ref.table("crime")



# API request - fetch the table

table = client.get_table(table_ref)
# Print information on all the columns in the "crime" table in the "hacker_news" dataset

table.schema
# Preview the first five lines of the "crime" table

client.list_rows(table, max_results=5).to_dataframe()
# Preview the first five entries in the "primary_type" column of the "crime" table

client.list_rows(table, selected_fields=table.schema[5:6], max_results=5).to_dataframe()