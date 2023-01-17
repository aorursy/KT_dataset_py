from google.cloud import bigquery
# Creating a 'client' object

client = bigquery.Client()
dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
table_ref = dataset_ref.table('full')



table = client.get_table(table_ref)
table.schema
# Comment table reference

table_ref_comment = dataset_ref.table('comments')

table_comment = client.get_table(table_ref_comment)
table_comment.schema
client.list_rows(table, max_results=5).to_dataframe()
# List 5 rows from

client.list_rows(table_comment,max_results=5).to_dataframe()