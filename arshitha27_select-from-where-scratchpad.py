from google.cloud import bigquery 



# creating a client object to communicate with bigquery

client = bigquery.Client()



# constructing a reference to "openaq" dataset

# reference to a dataset is like an address of the 

# dataset that's passed on to the client while fetching 

# actual dataset

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# using the dataset reference to fetch the dataset using

# the get_dataset method. Kinda like get_dataset is the 

# car and dataset_ref is the address. Client then would

# be the driver 



# API request 

dataset = client.get_dataset(dataset_ref)



# a dataset could have more than one table so, let's

# list them all

tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
# within the dataset we want to find the table

# therefore, we'd need its address and we get that 

# by constructing  a table reference to 

# global_air_quality

# it can be seen that we use the dataset reference to 

# construct the table ref. Analogy: dataset reference 

# is like the building address and table ref the apt 

# number

table_ref = dataset_ref.table("global_air_quality")



# API request - fetch the table

# here we rely on our previous driver: client

# but different car: get_table method 

table = client.get_table(table_ref)

type(table)



# preview the first five lines of the "global_air_quality"

# table

client.list_rows(table, max_results=5).to_dataframe()
table.schema
query = """

SELECT city FROM `bigquery-public-data.openaq.global_air_quality`

WHERE country='US'

"""
query_job = client.query(query)

type(query_job)
us_cities = query_job.to_dataframe()
us_cities.head()
us_cities['city'].value_counts().head()
# Query to get the score column from every row where the type column has value "job"

query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = "IN" 

        """



# Create a QueryJobConfig object to estimate size of query without running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)



# dry run is like a dress rehersal. Estimates are calculated

# without running the query. Learned in intro to databases, check it out! 

# could be an important interview question



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