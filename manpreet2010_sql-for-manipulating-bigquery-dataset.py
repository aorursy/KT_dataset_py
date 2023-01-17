from google.cloud import bigquery



client = bigquery.Client()

dataset_ref = client.dataset("openaq",project = "bigquery-public-data")



dataset =  client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))



for tb in tables:

    print(tb.table_id)

table_ref = dataset_ref.table("global_air_quality")

table = client.get_table(table_ref)

client.list_rows(table, max_results=5).to_dataframe()
table.schema
query = """ SELECT  city, pollutant, value from `bigquery-public-data.openaq.global_air_quality`

WHERE country = 'US' 

"""
# Create a "Client" object

client = bigquery.Client()
# Set up the query

query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()
us_cities.tail(5)
us_cities.city.value_counts().head(10)
query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
query_job = client.query(query)
query_result = query_job.to_dataframe()
query_result.head(5)
# Query to get the score column from every row where the type column has value "job"

query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US' 

        """



# Create a QueryJobConfig object to estimate size of query without running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)



# API request - dry run query to estimate costs

dry_run_query_job = client.query(query, job_config=dry_run_config)



print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
# Only run the query if it's less than 100 MB

Ten_MB = 100*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=Ten_MB)



# Set up the query (will only run if it's less than 100 MB)

safe_query_job = client.query(query, job_config=safe_config)



# API request - try to run the query, and return a pandas DataFrame

safe_query_job.to_dataframe()
ds