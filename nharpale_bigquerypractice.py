from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
table_ref = dataset_ref.table("comments")

table = client.get_table(table_ref)
table.schema
client.list_rows(table, max_results=5).to_dataframe()
import pandas as pd
bq_file = client.list_rows(table, max_results=25).to_dataframe()



#save file

with open("bqassignment1.csv", "w") as fi:

    fi.write(bq_file.to_csv())