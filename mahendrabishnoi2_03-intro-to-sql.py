from google.cloud import bigquery
# Create a client object

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# construct a reference to the hacker_news dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List tables present in "hacker_news" dataset

tables = list(client.list_tables(dataset))



# print names of all tables present in the dataset

for table in tables:

    print(table.table_id)
# Create a table reference ("full")

table_ref = dataset_ref.table("full")



# API request - fetch the full table

table = client.get_table(table_ref)
table.schema
# print first five rows of the "full" table

client.list_rows(table, max_results=5).to_dataframe()
# print first five rows of "by" column (or field) of "full" table

client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
# imports 

from google.cloud import bigquery



# create a client object 

client = bigquery.Client()



# create a dataset reference (to openaq)

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# api request - fetch data

dataset = client.get_dataset(dataset_ref)



# create a list of tables present in the dataset

tables = list(client.list_tables(dataset))



# print all table names

for table in tables:

    print(table.table_id)
# construct a reference to the table

table_ref = dataset_ref.table("global_air_quality")



# api request - fetch table

table = client.get_table(table_ref)



# print first five rows / lines of "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()
# Query to select all the items from the "city" column where the "country" column is 'US'

query = """

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
client = bigquery.Client()
query_job = client.query(query)
# api request - run the query and return a pandas DataFrame

us_cities = query_job.to_dataframe()
us_cities.city.value_counts().head()
query = """

        SELECT city, country

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country="US"

        """
query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country="US"

        """
# query to get the score column from every row where the type column has value "job"

query = """

        SELECT score, title

        FROM `bigquery-public-data.hacker_news.full`

        WHERE type = "job" 

        """



# create 'QueryJobConfig' object to estimate size of query without running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)



# api request - dry run query to estimate costs

dry_run_query_job = client.query(query, job_config=dry_run_config)



print("This query will process {} bytes".format(dry_run_query_job.total_bytes_processed))
def safe_config_f(max_size):

    return bigquery.QueryJobConfig(maximum_bytes_billed=max_size)
max_size = 1 

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=max_size)



# setup the query (only run if its less than 100 MB)

safe_query_job = client.query(query, job_config=safe_config)



# api request - run the query and return a pandas dataframe

safe_query_job.to_dataframe()





# strangely this query runs (need to take a look on documentation)
# Only run the query if it's less than 1 GB

ONE_GB = 1000*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_GB)



# Set up the query (will only run if it's less than 1 GB)

safe_query_job = client.query(query, job_config=safe_config)



# API request - try to run the query, and return a pandas DataFrame

job_post_scores = safe_query_job.to_dataframe()



# Print average score for job posts

job_post_scores.score.mean()
# imports 

from google.cloud import bigquery



# create a client object

client = bigquery.Client()



# construct a reference to the hacker_news dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# api request - fetch dataset

dataset = client.get_dataset(dataset_ref)



# list tables in the 'hacker_news' dataset

tables = list(client.list_tables(dataset))



# print all tables present in the dataset

for table in tables:

    print(table.table_id)
# construct a reference to 'comments' table

table_ref = dataset_ref.table('comments')



# api request - fetch dataset

table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()
# query to select comments that have more than 10 replies

query = """

        SELECT parent, COUNT(id)

        FROM `bigquery-public-data.hacker_news.comments`

        GROUP BY parent

        HAVING COUNT(id) > 10

        """
# set up query and QueryJobConfig (To be on safer side)

safe_config = bigquery.QueryJobConfig(max_bytes_billed = 10*9)       # 1 GB limit

query_job = client.query(query, job_config=safe_config)



# api request - run the query and return a pandas dataframe

popular_comments = query_job.to_dataframe()



# print first five rows of popular_comments dataframe

popular_comments.head()
# Imporoved version of earlier query with aliasing and improved readability

query_improved = """

                 SELECT parent, COUNT(1) AS NumPosts

                 FROM `bigquery-public-data.hacker_news.comments`

                 GROUP BY parent

                 HAVING COUNT(1) > 10

                 """



# set up query and QueryJobConfig (To be on safer side)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9) 

query_job = client.query(query_improved, job_config=safe_config)



# api request - run the query and return a pandas dataframe

improved_df = query_job.to_dataframe()



# print first five rows of improved_df dataframe

improved_df.head()
query_good = """

             SELECT parent, COUNT(id)

             FROM `bigquery-public-data.hacker_news.comments`

             GROUP BY parent

             """



# Example of a good query as parent is used with GROUP BY and id is used with COUNT()
query_bad = """

            SELECT author, parent, COUNT(id)

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY parent

            """



# This query will throw an error because author is neither aggregated nor used with GROUP BY
# imports

from google.cloud import bigquery



# create a client object

client = bigquery.Client()



# construct a reference to the 'nhtsa_traffic_fatalities' database

dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")



# api request - fetch dataset

dataset = client.get_dataset(dataset_ref)



# construct a reference to the 'accident_2015' table

table_ref = dataset_ref.table("accident_2015")



# api request - fetch the table

table = client.get_table(table_ref)



# show first five rows of the table fetched

client.list_rows(table, max_results=5).to_dataframe()
# query to find number of accidents on each day of the week

query = """

        SELECT EXTRACT(DAYOFWEEK from timestamp_of_crash) AS day_of_week,

               COUNT(consecutive_number) AS num_accidents

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week

        ORDER BY num_accidents DESC

        """
# set up query and QueryJobConfig

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(query, job_config=safe_config)



# run query and convert the result to a dataframe

df = query_job.to_dataframe()



df
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "crypto_bitcoin" dataset

dataset_ref = client.dataset("crypto_bitcoin", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "transactions" table

table_ref = dataset_ref.table("transactions")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "transactions" table

client.list_rows(table, max_results=5).to_dataframe()
# Query to select number of transactions per date, sorted by data

query = """

        WITH time AS

        (

        SELECT DATE(block_timestamp) AS trans_date

        FROM `bigquery-public-data.crypto_bitcoin.transactions`

        )

        SELECT COUNT(1) AS transactions,

               trans_date

        FROM time

        GROUP BY trans_date

        ORDER BY trans_date

        """
# set up query 

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# run query return dataframe

df = query_job.to_dataframe()



df.head()
df.set_index('trans_date').plot()
# imports

from google.cloud import bigquery



# create a client object

client = bigquery.Client()



# construct a reference to the 'github_repos' dataset

dataset_ref = client.dataset('github_repos', project="bigquery-public-data")



# api request - fetch dataset

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
# construct a reference to the 'licenses' table

licenses_table_ref = dataset_ref.table('licenses')



# api - request - fetch table

licenses_table = client.get_table(licenses_table_ref)



# show first five rows of table

client.list_rows(licenses_table, max_results=5).to_dataframe()
# construct a reference to the 'sample_files' table

sample_files_table_ref = dataset_ref.table('sample_files')



# api request - fetch table

sample_files_table = client.get_table(sample_files_table_ref)



# show first five rows of sample_files table

client.list_rows(sample_files_table, max_results=5).to_dataframe()
# Query to determine the number of files per license, sorted by number of files

query = """

        SELECT L.license, COUNT(1) AS num_files

        FROM `bigquery-public-data.github_repos.sample_files` AS sf

        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L

            ON sf.repo_name = L.repo_name

        GROUP BY L.license

        ORDER BY num_files DESC

        """



# set up query 

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# run query return dataframe

file_count_by_license = query_job.to_dataframe()



file_count_by_license.head()