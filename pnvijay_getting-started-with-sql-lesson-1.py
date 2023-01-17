from google.cloud import bigquery
# Create a "Client" Object

client = bigquery.Client()
# Create a reference to the hackernews dataset

dataset_ref = client.dataset("hacker_news",project='bigquery-public-data')



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in client dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in dataset

for table in tables:

    print(table.table_id)
# Construct a reference to the "full" table

table_ref = dataset_ref.table('full')



# API Request - fetch the table

table = client.get_table(table_ref)
# List the table schema

# Schema has name of column, type of data in column, can it have null values or not, description of data in column

table.schema
# List the five rows of the table

client.list_rows(table,max_results=5).to_dataframe()
# List the five entries in the "by" column of the 'full' table

client.list_rows(table,selected_fields=table.schema[:1],max_results=5).to_dataframe()