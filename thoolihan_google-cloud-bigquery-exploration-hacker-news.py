from google.cloud import bigquery

PROJECT_ID = 'bigquery-public-data'

bq_client = bigquery.Client(project=PROJECT_ID)



from google.cloud import storage

PROJECT_ID = 'big-query-tutorial-259901'

st_client = storage.Client(project=PROJECT_ID)
ds_ref = bq_client.dataset("hacker_news")

dataset = bq_client.get_dataset(ds_ref)
for table in bq_client.list_tables(dataset):

    print(table.table_id)
tb_ref = ds_ref.table("full")

table = bq_client.get_table(tb_ref)
table.schema
bq_client.list_rows(table, max_results=5).to_dataframe()
gb = table.num_bytes / (1024 ** 3)

print("Table has {:,d} rows and is {:,.3f} GB".format(table.num_rows, gb))