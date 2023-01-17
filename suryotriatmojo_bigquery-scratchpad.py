from google.cloud import bigquery
# Create a "Client" object

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

print(dataset_ref)

print(type(dataset_ref))



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)

print(dataset)

print(type(dataset))
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))

print(tables)

print(type(tables))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "full" table

table_ref = dataset_ref.table("full")

print(table_ref)

print(type(table_ref))



# API request - fetch the table

table = client.get_table(table_ref)

print(table)

print(type(table))
# Print information on all the columns in the "full" table in the "hacker_news" dataset

table.schema
# Preview the first five lines of the "full" table

client.list_rows(table, max_results=5).to_dataframe()
# Preview the first five entries in the "by" column of the "full" table

client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
# Preview the first five lines of the "full" table

bq_file = client.list_rows(table, max_results=25).to_dataframe()



# save file 

with open("my_file.csv", "w") as file:

    file.write(bq_file.to_csv())
# link to all the BigQuery client commands

# https://googleapis.github.io/google-cloud-python/latest/bigquery/reference.html