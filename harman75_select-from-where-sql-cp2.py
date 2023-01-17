from google.cloud import bigquery
client =bigquery.Client()
dataset_ref=client.dataset("openaq",project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))

for table in tables:  

    print(table.table_id)
#Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("global_air_quality")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=1).to_dataframe()
query = """

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
client = bigquery.Client()
query_job = client.query(query)
us_cities = query_job.to_dataframe()
us_cities.city.value_counts().head()