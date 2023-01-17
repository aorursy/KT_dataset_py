from google.cloud import bigquery

client = bigquery.Client()
dataset_ref = client.dataset("openaq", project = "bigquery-public-data")

dataset = client.get_dataset(dataset_ref)

tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)
table_ref = dataset_ref.table("global_air_quality")

table = client.get_table(table_ref)

client.list_rows(table, max_results=5 ).to_dataframe()
table.schema
query = """

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
query_job = client.query(query)
us_cities = query_job.to_dataframe()
us_cities
us_cities.city.value_counts().head()
query = """

        SELECT city , source_name

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """

query_job = client.query(query)

us_cities = query_job.to_dataframe()

us_cities.head()
query = """

        SELECT * 

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """

query_job = client.query(query)

us_cities = query_job.to_dataframe()

us_cities.head()
query = """

        SELECT * 

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'IN'

        """

# Create a QueryJobConfig object to estimate size of query without running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)



# API request - dry run query to estimate costs

dry_run_query_job = client.query(query, job_config=dry_run_config)



print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
ONE_HUNDRED_MB = 100*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_HUNDRED_MB)



# Set up the query (will only run if it's less than 100 MB)

safe_query_job = client.query(query, job_config=safe_config)



# API request - try to run the query, and return a pandas DataFrame

safe_query_job.to_dataframe()