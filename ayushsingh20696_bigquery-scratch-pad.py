from google.cloud import bigquery

import pandas as pd
client = bigquery.Client()
# Constuct a reference to dataset "hacker_news"

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request fetch the dataset

dataset=client.get_dataset(dataset_ref)
tables=list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)
table_ref=dataset_ref.table("comments")

table=client.get_table(table_ref)
table.schema
client.list_rows(table, max_results=5).to_dataframe()
client.list_rows(table,selected_fields=table.schema[:6], max_results=5).to_dataframe()
bq_file=client.list_rows(table,max_results=25).to_dataframe()



with open("my_file.csv","w") as file:

    file.write(bq_file.to_csv())