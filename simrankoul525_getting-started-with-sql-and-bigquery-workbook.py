from google.cloud import bigquery
# Creating a client object

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "full" table

table_ref = dataset_ref.table("crime")



# API request - fetch the table

table = client.get_table(table_ref)
table.schema
# Preview the first five lines of the "full" table

client.list_rows(table, max_results=5).to_dataframe()
# FIRST FIVE COLUMNS and first 10 values 

client.list_rows(table, selected_fields=table.schema[:5], max_results=10).to_dataframe()

## previous commands

import bq_helper

# create a helper object for our bigquery dataset

chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",

                            

                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema('crime')
chicago_crime.head('crime')