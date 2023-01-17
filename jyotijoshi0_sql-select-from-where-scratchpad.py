from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "openaq" dataset

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# List all the tables in the "openaq" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there's only one!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("global_air_quality")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=10).to_dataframe()
table.schema
# Query to select all the items from the "city" column where the "country" column is 'US'

query = """

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
# Set up the query

query_job = client.query(query)



# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()
us_cities
# What five cities have the most measurements?

us_cities.city.value_counts().head()
query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """



# Set up the query

query_job = client.query(query)



# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()



us_cities.head()
# Query to get the score column from every row where the type column has value "job"

query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country LIKE "I%" 

        """



# Create a QueryJobConfig object to estimate size of query without running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)



# API request - dry run query to estimate costs

dry_run_query_job = client.query(query, job_config=dry_run_config)



print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
# Only run the query if it's less than 100 MB

ONE_HUNDRED_MB = 100*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_HUNDRED_MB)



# Set up the query (will only run if it's less than 100 MB)

safe_query_job = client.query(query, job_config=safe_config)



# API request - try to run the query, and return a pandas DataFrame

safe_query_job.to_dataframe()
# returns pollutants and a new column if they have number at the end

query = """

        SELECT pollutant,

            REGEXP_CONTAINS(pollutant, r'[0-9]+') AS has_number    

        FROM `bigquery-public-data.openaq.global_air_quality`

        """



# Set up the query

query_job = client.query(query)



# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()



us_cities.head()
# returns country and a new coulmn with if they starts with S

query = """

        SELECT country,

            REGEXP_CONTAINS(country, r'^[Ss].*') AS starts_with_s 

        FROM `bigquery-public-data.openaq.global_air_quality`

        """



# Set up the query

query_job = client.query(query)



# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()



us_cities.head()