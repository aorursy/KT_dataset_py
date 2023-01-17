from google.cloud import bigquery
client = bigquery.Client()



dataset_ref = client.dataset("openaq", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))

for table in tables:

    print (table.table_id)



table_ref = dataset_ref.table("global_air_quality")

table = client.get_table(table_ref)
table.schema
df = client.list_rows(table, max_results=5).to_dataframe()

df
# Query to select all the items from the "city" column where the "country" column is 'US'

Query = """ 

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = "US"

        """
#setup the query

query_job = client.query(Query)
# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()
us_cities.city.value_counts().head()
# Query to get the score column from every row where the type column has value "job"

Query = """ 

        SELECT score, title

        FROM `bigquery-public-data.hacker_news.full`

        WHERE type="job"

        """



# Create a QueryJobConfig object to estimate size of query without running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)



# API request - dry run query to estimate costs

dry_run_query_job = client.query(Query, job_config=dry_run_config )



print("This Query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))

# Only run the query if it's less than 100 MB

ONE_HUNDRED_MB = 100*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_HUNDRED_MB)



# Set up the query (will only run if it's less than 100 MB)

safe_run_job = client.query(Query, job_config = safe_config)



# API request - try to run the query, and return a pandas DataFrame

safe_run_job.to_dataframe()

# Only run the query if it's less than 1 GB

ONE_GB = 1000*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_GB)



# Set up the query (will only run if it's less than 100 MB)

safe_run_job = client.query(Query, job_config = safe_config)



# API request - try to run the query, and return a pandas DataFrame

job_post_scores = safe_run_job.to_dataframe()



# Print average score for job posts

job_post_scores.score.mean()
# Query to select comments that received more than 10 replies

query_popular = """

                SELECT parent, COUNT(id)

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                """
# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_popular, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

popular_comments = query_job.to_dataframe()



popular_comments.head()



# Improved version of earlier query, now with aliasing & improved readability

query_improved = """

                SELECT parent, COUNT(id) AS NumPosts

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_improved, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

popular_comments = query_job.to_dataframe()



popular_comments.head()
## working with Dates

client = bigquery.Client()

# Construct a reference to the "nhtsa_traffic_fatalities" dataset

dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")

#API request

dataset = client.get_dataset(dataset_ref)



tables = client.list_tables(dataset)



for t in tables:

    print (t.table_id)
table_ref = dataset_ref.table("accident_2015")

table = client.get_table(table_ref)
table.schema
client.list_rows(table, max_results=5).to_dataframe()
# Query to find out the number of accidents for each day of the week

Query = """

        SELECT COUNT(consecutive_number) AS num_accidents,

               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week

        ORDER BY num_accidents desc

       """
# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(Query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

accidents_by_day = query_job.to_dataframe()

accidents_by_day
client = bigquery.Client()



dataset_ref = client.dataset("crypto_bitcoin", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))

for table in tables:

    print (table.table_id)



table_ref = dataset_ref.table("transactions")

table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()
# Query to select the number of transactions per date, sorted by date

query_with_CTE = """ 

                 WITH time AS

                 (

                     SELECT DATE(block_timestamp) as trans_date

                     FROM `bigquery-public-data.crypto_bitcoin.transactions`

                 )

                 SELECT count(1) AS transactions, trans_date

                 FROM time

                 GROUP BY trans_date

                 ORDER BY trans_date

                 

                """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_with_CTE, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

transactions_by_date = query_job.to_dataframe()



# Print the first five rows

transactions_by_date.head()
transactions_by_date.set_index('trans_date').plot()
client = bigquery.Client()

dataset_ref = client.dataset("github_repos", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



#tables = list(client.list_tables(dataset))

#for t in tables:

#    print(t.table_id)



table_ref = dataset_ref.table('licenses')

table = client.get_table(table_ref)



client.list_rows(table, max_results=5).to_dataframe()
table_ref2 = dataset_ref.table('sample_files')

table2 = client.get_table(table_ref2)

client.list_rows(table2, max_results=5).to_dataframe()
# Query to determine the number of files per license, sorted by number of files

Query = """

        SELECT L.license, count(1) as num_of_files

        FROM `bigquery-public-data.github_repos.sample_files` as SF

        INNER JOIN `bigquery-public-data.github_repos.licenses` as L

               ON L.repo_name = SF.repo_name

        GROUP BY L.license

        ORDER BY num_of_files

        """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(Query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

file_count_by_license = query_job.to_dataframe()
file_count_by_license