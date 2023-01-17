from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "nhtsa_traffic_fatalities" dataset

dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



#MYWORK

tables = list(client.list_tables(dataset))

#for table in tables:

    #print(table.table_id)



# Construct a reference to the "accident_2015" table

table_ref = dataset_ref.table("accident_2015")







# API request - fetch the table

table = client.get_table(table_ref)



#table.schema



# Preview the first five lines of the "accident_2015" table

client.list_rows(table, max_results=5).to_dataframe()
query = """

            SELECT *

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            """

client = bigquery.Client()

dry_run_config = bigquery.QueryJobConfig(dry_run=True)

dry_run_query_job = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))

            